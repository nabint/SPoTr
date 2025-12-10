import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from openpoints.models import build_model_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.utils.config import EasyConfig
from openpoints.utils import load_checkpoint


from tree_seg_spotr.spotr_train_dataloader import TreePartNormalDatasetLargeNumPoints

import provider

seg_classes = {"standing_trees": [0], "fallen_trees": [1, 2, 3]}


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


# Data utilities
def _augment_positions_numpy(pos):
    pos = provider.random_scale_point_cloud(pos)
    pos = provider.shift_point_cloud(pos)
    return pos


def build_model_input_from_batch(
    points,
    label,
    target,
    augment,
    device,
):
    pos_np = points[:, :, :3].cpu().numpy()

    if augment:
        pos_np = _augment_positions_numpy(pos_np)

    pos = torch.from_numpy(pos_np).float().to(device)

    feats = points[:, :, 3:].to(device)

    x = torch.cat([pos, feats], dim=2)

    data = {
        "pos": pos.contiguous(),  # [B, N, 3]
        "x": x.permute(0, 2, 1).contiguous(),  # [B, C_feat=7, N]
        "cls": label,
        "y": target,
    }

    return data


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    num_classes,
    num_part,
    augment,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    batch_count = 0

    for points, label, target, data_chosen, filename in train_loader:
        batch_count += 1
        optimizer.zero_grad()

        points = points.float().to(device)
        label = label.long().to(device)
        target = target.long().to(device)

        # Batch, number of points, no of Channels
        B, N, C = points.shape

        # build model input (including augmentation if requested)
        data = build_model_input_from_batch(
            points,
            label,
            target,
            augment=augment,
            device=device,
        )

        # Returns Batch, num_classes, points
        logits = model(data)

        # print(logits.shape)
        # exit()

        _, P, _ = logits.shape

        # flatten for loss / preds
        # First convert to Batch, no of points, num_class
        # and then Collapse earlier and keep only P
        logits_flat = logits.permute(0, 2, 1).reshape(-1, P)  # [B*N, P]
        target_flat = target.view(-1)  # [B*N]

        if "MultiShapeCrossEntropy" in str(criterion):
            loss = criterion(logits_flat, target_flat, data["cls"])
        else:
            loss = criterion(logits_flat, target_flat)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        preds = logits_flat.max(1)[1]

        total_correct += (preds == target_flat).sum().item()
        total_points += target_flat.numel()

        # cur_loss = total_loss / (batch_count * train_loader.batch_size)
        # cur_acc = total_correct / float(total_points) if total_points > 0 else 0.0
        # pbar.set_postfix({"loss": f"{cur_loss:.6f}", "acc": f"{cur_acc:.6f}"})

        optimizer.zero_grad()

    avg_loss = (
        total_loss / float(len(train_loader.dataset))
        if len(train_loader.dataset) > 0
        else 0.0
    )
    avg_acc = total_correct / float(total_points) if total_points > 0 else 0.0
    return avg_loss, avg_acc


def evaluate_openpoints(
    model,
    test_loader,
    device,
    seg_classes,
    num_part,
    num_classes,
):
    model.eval()

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(num_part)]
    total_correct_class = [0 for _ in range(num_part)]
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}
    pred_1_correct = []

    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    # print("Seg lablel to cat", seg_label_to_cat)

    accuracy_combination = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), disable=True)
        for batch_id, (points, label, target, data_chosen, filename, *rest) in pbar:
            points = points.float().to(device)
            label = label.long().to(device)
            target = target.long().to(device)

            B, N, C = points.shape

            data = build_model_input_from_batch(points, label, target, False, device)

            # Returns Batch, num_classes, points
            logits = model(data)

            cur_pred_val_logits = (
                logits.cpu().data.numpy()
            )  # [Batch, Parts, No of points]

            # print("Current pred logits", cur_pred_val_logits.shape)

            cur_pred_val = np.zeros((B, N)).astype(np.int32)
            target_np = target.cpu().data.numpy()

            # print("Target Np:", target_np)
            # print("Target Np shape", target_np.shape)

            # print(target_np.ndim)

            # print(cur_pred_val_logits)
            # print("-----------------")
            # print(np.unique(target_np))
            # print(target_np.count(0))
            # exit()

            # print("Current pred value shape", cur_pred_val.shape)
            # print("Current pred dimesino", cur_pred_val.ndim)

            # For each sample in batch: restrict to parts relevant to its category
            for i in range(B):
                cat = seg_label_to_cat[target_np[0, i]]  # gives standing or fallen

                logits_i = cur_pred_val_logits[i, :, :]  # [P, N]

                parts = seg_classes[
                    cat
                ]  # indices of part labels for this category, either [0 or [1,2,3]]

                # print("cat", cat)
                # print("Seg clas", seg_classes)

                # Take logits only at those part indices, then argmax
                sub_logits = logits_i[parts, :].transpose(1, 0)  # [N, num_cat_parts]

                # print("Logits shape", logits_i.shape)
                # print("Sub Logits shape", sub_logits.shape)

                sub_preds = np.argmax(sub_logits, axis=1)
                # print("Parts", parts)
                # print("Sub Preds. Parts", sub_preds)
                # print("After Sub Preds. Parts", sub_preds + parts[0])
                # print("TArget ", target_np)

                cur_pred_val[i, :] = sub_preds + parts[0]
                # print(
                #     "pred",
                #     np.unique(cur_pred_val[i, :]),
                #     "real",
                #     np.unique(target_np[i, :]),
                # )

            # print("Current pred val", cur_pred_val.shape)
            # accuracy counting (per-point)
            correct = np.sum(cur_pred_val == target_np)
            total_correct += correct
            total_seen += B * N

            # print("correct", correct)
            accuracy_combination += correct / (B * N)

            # per-class counts
            for l in range(num_part):
                total_seen_class[l] += np.sum(target_np == l)
                total_correct_class[l] += np.sum((cur_pred_val == l) & (target_np == l))

            # compute IoU per sample
            for i in range(B):
                segp = cur_pred_val[i, :]
                segl = target_np[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum(
                            (segl == l) & (segp == l)
                        ) / float(np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

    # aggregate IoUs & metrics
    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat]) if len(shape_ious[cat]) > 0 else 0.0
    mean_shape_ious = np.mean(list(shape_ious.values())) if len(shape_ious) > 0 else 0.0
    accuracy = total_correct / float(total_seen) if total_seen > 0 else 0.0
    class_avg_accuracy = np.mean(
        np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64)
    )
    instance_avg_iou = np.mean(all_shape_ious) if len(all_shape_ious) > 0 else 0.0

    print(pred_1_correct)
    print(np.sum(pred_1_correct))

    test_metrics = {
        "accuracy": accuracy,
        "class_avg_accuracy": class_avg_accuracy,
        "class_avg_iou": mean_shape_ious,
        "instance_avg_iou": instance_avg_iou,
        "per_category_iou": shape_ious,
    }
    return test_metrics


def init_datasets(cfg):
    # Dataset & loaders
    TRAIN_DATASET = TreePartNormalDatasetLargeNumPoints(
        root=cfg.dataset.train_root, split="trainval", use_rgbi=True
    )
    TEST_DATASET = TreePartNormalDatasetLargeNumPoints(
        root=cfg.dataset.train_root, split="test", use_rgbi=True, is_test=True
    )

    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=5,
        shuffle=True,
        num_workers=5,
        drop_last=True,
    )
    test_loader = DataLoader(
        TEST_DATASET,
        batch_size=1,
        shuffle=False,
        num_workers=5,
    )
    return train_loader, test_loader


def main_training_loop(
    cfg: EasyConfig,
    seg_classes: dict,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = build_model_from_cfg(cfg.model)

    # Load the pretrained checkpoint
    checkpoint = torch.load(cfg.pretrained_path, map_location="cpu", weights_only=False)

    # # Extract state dict
    if "model" in checkpoint:
        pretrained_state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        pretrained_state_dict = checkpoint["state_dict"]
    else:
        pretrained_state_dict = checkpoint

    # Filter out the mismatched layers
    model_state_dict = model.state_dict()
    filtered_state_dict = {}

    for k, v in pretrained_state_dict.items():
        if k in model_state_dict:
            # Only load if shapes match
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(
                    f"Skipping {k}: pretrained shape {v.shape} != model shape {model_state_dict[k].shape}"
                )
        else:
            print(f"Skipping {k}: not found in new model")

    # # Load the filtered weights (strict=False allows missing keys)
    model.load_state_dict(filtered_state_dict, strict=False)

    # checkpoint = torch.load(cfg.pretrained_path, map_location='cpu', weights_only=False)
    # pretrained_state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))

    # model.load_state_dict(pretrained_state_dict, strict=False)

    # load_checkpoint(model, pretrained_path=cfg.pretrained_path)

    model.apply(inplace_relu)
    model.to(device)

    criterion = build_criterion_from_cfg(cfg.criterion_args).to(device)

    # Optimizer selection
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=0.9,
        )

    # Learning / BN schedule params
    LR_CLIP = 1e-5
    MOMENTUM_ORIG = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = 1

    best_acc = 0.0
    best_class_avg_iou = 0.0
    best_instance_avg_iou = 0.0

    # Logging & checkpoint directory
    exp_dir = Path(cfg.weight_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Training Code:
    for epoch in range(cfg.epochs):
        lr = max(cfg.learning_rate * (cfg.lr_decay ** (epoch // 20)), LR_CLIP)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        momentum = MOMENTUM_ORIG * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01

        def _bn_adjust(m):
            if isinstance(m, torch.nn.BatchNorm1d) or isinstance(
                m, torch.nn.BatchNorm2d
            ):
                m.momentum = momentum

        model.apply(_bn_adjust)

        train_loader, test_loader = init_datasets(cfg)

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            cfg.num_classes,
            cfg.num_part,
            augment=True,
        )
        print(
            f"Epoch {epoch + 1}/{cfg.epochs}  Train Loss: {train_loss:.6f}  Train Acc: {train_acc:.6f}"
        )

        # Evaluate after epoch
        test_metrics = evaluate_openpoints(
            model, test_loader, device, seg_classes, cfg.num_part, cfg.num_classes
        )
        print(
            "Test Acc: {:.6f}  Class mIoU: {:.6f}   mIoU: {:.6f}".format(
                test_metrics["accuracy"],
                test_metrics["class_avg_iou"],
                test_metrics["instance_avg_iou"],
            )
        )

        # Save best by instance mIoU (same rule)
        if test_metrics["instance_avg_iou"] >= best_instance_avg_iou:
            savepath = str(exp_dir / "optimized_best_model.pth")
            print("Saving model to", savepath)
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "test_metrics": test_metrics,
            }
            torch.save(state, savepath)
            best_instance_avg_iou = test_metrics["instance_avg_iou"]

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
        if test_metrics["class_avg_iou"] > best_class_avg_iou:
            best_class_avg_iou = test_metrics["class_avg_iou"]

        print(
            "Best acc: {:.6f}  Best class mIoU: {:.6f}  Best instance mIoU: {:.6f}".format(
                best_acc, best_class_avg_iou, best_instance_avg_iou
            )
        )
        # break

    print("Training finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="config yaml")
    parser.add_argument("--batch_size", type=int, default=5)

    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--weight_dir", type=str, default="./weights")
    parser.add_argument("--decay_rate", type=float, default=1e-4)

    parser.add_argument("--npoint", type=int, default=2048)
    parser.add_argument("--normal", action="store_true", default=False)

    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--data_root", type=str, default="./data")

    args = parser.parse_args()

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    args_dict = vars(args)
    for key, value in args_dict.items():
        if key != "cfg":
            cfg[key] = value

    main_training_loop(cfg, seg_classes)


if __name__ == "__main__":
    main()
