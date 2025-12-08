#!/usr/bin/env python3
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from openpoints.models import build_model_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.utils.config import EasyConfig
from train.spotr_train_dataloader import TreePartNormalDatasetLargeNumPoints


from pathlib import Path
from tqdm import tqdm

import numpy as np

import provider  # your augmentation helpers (random_scale_point_cloud, shift_point_cloud)


# -------------------------
# Helper: to_categorical (if you don't have it)
# -------------------------
def to_categorical(label, num_classes):
    # label: [B] or [B,1]
    if label.dim() == 2:
        label = label[:, 0]
    onehot = torch.zeros(label.size(0), num_classes, device=label.device)
    onehot.scatter_(1, label.view(-1, 1), 1)
    return onehot


# -------------------------
# Train one epoch (OpenPoints input format)
# -------------------------
def train_one_epoch_openpoints(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    num_classes,
    num_part,
    augment=True,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    batch_count = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train")
    for idx, (points, label, target, data_chosen, filename) in pbar:
        batch_count += 1
        optimizer.zero_grad()

        # points: [B, N, C] (C >= 3: x,y,z,(normals),(rgbi))
        # label: [B, 1]  or [B]
        # target: [B, N] segmentation labels

        # Move to device
        points = points.float().to(device)  # [B, N, C]
        label = label.long().to(device)  # [B, 1] or [B]
        target = target.long().to(device)  # [B, N]

        B, N, C = points.shape

        # Separate XYZ and features for augmentation
        pos = (
            points[:, :, :3].cpu().numpy()
        )  # do augmentations in numpy (provider uses numpy)
        feats = (
            points[:, :, 3:].to(device)
            if C > 3
            else torch.zeros(B, N, 0, device=device)
        )

        if augment:
            # reference uses provider.random_scale_point_cloud and shift_point_cloud
            pos = provider.random_scale_point_cloud(pos)
            pos = provider.shift_point_cloud(pos)
        pos = torch.from_numpy(pos).float().to(device)  # [B, N, 3]

        # Ensure feats has exactly 4 channels (pad or truncate)
        if feats.numel() == 0:
            # no extra features in input: create zeros (B,N,4)
            feats = torch.zeros(B, N, 4, device=device)
        else:
            if feats.shape[2] < 4:
                pad = torch.zeros(B, N, 4 - feats.shape[2], device=device)
                feats = torch.cat([feats, pad], dim=2)
            elif feats.shape[2] > 4:
                feats = feats[:, :, :4]

        # Build model input: keep OpenPoints format
        # model likely expects data["pos"]: [B, N, 3], data["x"]: [B, C_feat, N] (channels first)
        x = torch.cat([pos, feats], dim=2)  # [B, N, 7] typically
        data = {
            "pos": pos,  # [B, N, 3]
            "x": x.permute(0, 2, 1).contiguous(),  # [B, C_feat, N]
            "cls": label.view(B, -1) if label.dim() == 2 else label.view(B, 1),
            "y": target,  # keep target for convenience if needed
        }

        data["pos"] = data["pos"].contiguous()
        data["x"] = data["x"].contiguous()

        # Forward pass (OpenPoints style)
        logits = model(data)  # expected shape [B, P, N]

        # If some models return dicts, we already agreed A: plain tensor
        if logits.dim() == 4:  # defensive: squeeze if extra dims
            logits = logits.squeeze(0)
        if logits.dim() != 3:
            raise ValueError(
                f"Model output logits must be [B, P, N], got {logits.shape}"
            )

        # Bring logits to shape [B, P, N] -> produce logits_flat [B*N, P]
        B_out, P, N_out = logits.shape
        assert B_out == B and N_out == N, (
            "Mismatch between model output and input sizes"
        )

        logits_flat = logits.permute(0, 2, 1).reshape(-1, P)  # [B*N, P]
        target_flat = target.view(-1)  # [B*N]

        # Compute loss (supporting MultiShapeCrossEntropy that needs cls)
        if getattr(criterion, "NAME", None) == "MultiShapeCrossEntropy":
            loss = criterion(logits_flat, target_flat, data["cls"].view(-1))
        else:
            loss = criterion(logits_flat, target_flat)

        loss.backward()
        optimizer.step()

        # Metrics (batch)
        total_loss += loss.item() * B
        preds = logits_flat.max(1)[1]  # [B*N]
        total_correct += (preds == target_flat).sum().item()
        total_points += target_flat.numel()

        cur_loss = total_loss / (batch_count * train_loader.batch_size)
        cur_acc = total_correct / float(total_points)
        pbar.set_postfix({"loss": f"{cur_loss:.4f}", "acc": f"{cur_acc:.4f}"})

    avg_loss = total_loss / float(len(train_loader.dataset))
    avg_acc = total_correct / float(total_points)
    return avg_loss, avg_acc


# -------------------------
# Eval function (matches reference calculation)
# -------------------------
def evaluate_openpoints(model, test_loader, device, seg_classes, num_part, num_classes):
    model.eval()
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(num_part)]
    total_correct_class = [0 for _ in range(num_part)]
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Eval")
        for batch_id, (points, label, target, data_chosen, filename, *rest) in pbar:
            # points shape [B, N, C]
            points = points.float().to(device)
            label = label.long().to(device)
            target = target.long().to(device)

            B, N, C = points.shape
            pos = points[:, :, :3]  # do not augment in eval
            feats = points[:, :, 3:] if C > 3 else torch.zeros(B, N, 4, device=device)
            if feats.numel() == 0:
                feats = torch.zeros(B, N, 4, device=device)
            else:
                if feats.shape[2] < 4:
                    pad = torch.zeros(B, N, 4 - feats.shape[2], device=device)
                    feats = torch.cat([feats, pad], dim=2)
                elif feats.shape[2] > 4:
                    feats = feats[:, :, :4]

            x = torch.cat([pos, feats], dim=2)
            pos = pos.contiguous()
            x = x.contiguous()
            data = {
                "pos": pos,
                "x": x.permute(0, 2, 1).contiguous(),
                "cls": label.view(B, -1) if label.dim() == 2 else label.view(B, 1),
            }

            logits = model(data)  # [B, P, N]
            cur_pred_val_logits = logits.cpu().data.numpy()  # [B, P, N]
            cur_pred_val = np.zeros((B, N)).astype(np.int32)
            target_np = target.cpu().data.numpy()

            # Per sample: select only parts relevant to category
            for i in range(B):
                cat = seg_label_to_cat[target_np[i, 0]]
                logits_i = cur_pred_val_logits[i, :, :]  # [P, N]
                # take only indices of seg_classes[cat]
                parts = seg_classes[cat]
                # logits_i shape [P, N] where P covers all classes. We need argmax across relevant part indices
                sub_logits = logits_i[parts, :].transpose(1, 0)  # [N, num_cat_parts]
                sub_preds = np.argmax(sub_logits, axis=1)
                cur_pred_val[i, :] = sub_preds + parts[0]

            # Accuracy counting (like reference)
            correct = np.sum(cur_pred_val == target_np)
            total_correct += correct
            total_seen += B * N

            for l in range(num_part):
                total_seen_class[l] += np.sum(target_np == l)
                total_correct_class[l] += np.sum((cur_pred_val == l) & (target_np == l))

            # IoU per sample
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

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat]) if len(shape_ious[cat]) > 0 else 0.0
    mean_shape_ious = np.mean(list(shape_ious.values()))
    accuracy = total_correct / float(total_seen) if total_seen > 0 else 0.0
    class_avg_accuracy = np.mean(
        np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64)
    )
    instance_avg_iou = np.mean(all_shape_ious) if len(all_shape_ious) > 0 else 0.0

    test_metrics = {
        "accuracy": accuracy,
        "class_avg_accuracy": class_avg_accuracy,
        "class_avg_iou": mean_shape_ious,
        "instance_avg_iou": instance_avg_iou,
        "per_category_iou": shape_ious,
    }
    return test_metrics


# -------------------------
# Main training loop (integrate with your existing main)
# -------------------------
def main_training_loop(cfg, seg_classes):
    # cfg: config object containing dataset paths, lr, epochs, etc.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build model & criterion (OpenPoints)
    model = build_model_from_cfg(cfg.model)
    model.to(device)
    criterion = build_criterion_from_cfg(cfg.criterion_args).to(device)

    # Dataset & loaders
    TRAIN_DATASET = TreePartNormalDatasetLargeNumPoints(
        root=cfg.dataset.train_root, split="trainval", use_rgbi=True
    )
    TEST_DATASET = TreePartNormalDatasetLargeNumPoints(
        root=cfg.dataset.train_root,
        split="test",
        use_rgbi=True,
        is_test=True,
    )

    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=1,
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

    # Optimizer
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
            model.parameters(), lr=cfg.learning_rate, momentum=0.9
        )

    # Learning schedule params
    LR_CLIP = 1e-5
    MOMENTUM_ORIG = 0.1
    MOMENTUM_DECAY = 0.5
    MOMENTUM_DECAY_STEP = 1

    best_acc = 0.0
    best_class_avg_iou = 0.0
    best_instance_avg_iou = 0.0

    # logging + checkpoint dir
    exp_dir = Path(cfg.log_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir.joinpath("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    for epoch in range(cfg.epochs):
        # lr schedule
        lr = max(cfg.learning_rate * (cfg.lr_decay ** (epoch // 20)), LR_CLIP)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # bn momentum update (if model uses BatchNorm1d/2d)
        momentum = MOMENTUM_ORIG * (MOMENTUM_DECAY ** (epoch // MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01

        def bn_adjust(m):
            if isinstance(m, torch.nn.BatchNorm1d) or isinstance(
                m, torch.nn.BatchNorm2d
            ):
                m.momentum = momentum

        model.apply(lambda m: bn_adjust(m))

        # Train
        train_loss, train_acc = train_one_epoch_openpoints(
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

        # Evaluate
        test_metrics = evaluate_openpoints(
            model, test_loader, device, seg_classes, cfg.num_part, cfg.num_classes
        )
        print(
            "Test Acc: {:.6f}  Class mIoU: {:.6f}  Instance mIoU: {:.6f}".format(
                test_metrics["accuracy"],
                test_metrics["class_avg_iou"],
                test_metrics["instance_avg_iou"],
            )
        )

        # Save best (by instance mIoU, like reference)
        if test_metrics["instance_avg_iou"] >= best_instance_avg_iou:
            savepath = str(checkpoints_dir / "best_model.pth")
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

        # track bests
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
        if test_metrics["class_avg_iou"] > best_class_avg_iou:
            best_class_avg_iou = test_metrics["class_avg_iou"]

        print(
            "Best acc: {:.6f}  Best class mIoU: {:.6f}  Best instance mIoU: {:.6f}".format(
                best_acc, best_class_avg_iou, best_instance_avg_iou
            )
        )

    print("Training finished.")


def apply_args_to_cfg(cfg, args):
    """
    If a value does NOT exist in cfg, fill it from args.
    If it exists in cfg, keep cfg's value.
    """

    # simple scalar overrides
    scalar_fields = [
        "model",
        "batch_size",
        "epoch",
        "learning_rate",
        "optimizer",
        "log_dir",
        "decay_rate",
        "npoint",
        "step_size",
        "lr_decay",
        "tag",
        "data_root",
    ]

    for field in scalar_fields:
        if not hasattr(cfg, field) or getattr(cfg, field) is None:
            setattr(cfg, field, getattr(args, field))

    # boolean flags
    if not hasattr(cfg, "normal"):
        cfg.normal = args.normal
    if not hasattr(cfg, "rgbi_channel"):
        cfg.rgbi_channel = args.rgbi_channel

    # GPU
    if not hasattr(cfg, "gpu"):
        cfg.gpu = args.gpu

    # Defaults required by training loop if YAML misses them
    if not hasattr(cfg, "num_classes"):
        cfg.num_classes = 4  # standing(1) + fallen(3)
    if not hasattr(cfg, "num_part"):
        cfg.num_part = 4  # 4 labels total
    if not hasattr(cfg, "epochs"):
        cfg.epochs = args.epoch

    # Dataset section
    if not hasattr(cfg, "dataset"):
        cfg.dataset = EasyConfig()

    if not hasattr(cfg.dataset, "train_root"):
        cfg.dataset.train_root = args.data_root

    return cfg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, required=True, help="config yaml")

    parser.add_argument("--model", type=str, default="pointnet_part_seg")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--npoint", type=int, default=2048)
    parser.add_argument("--normal", action="store_true", default=False)
    parser.add_argument("--rgbi_channel", action="store_true", default=False)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--data_root", type=str, default="./data")

    args = parser.parse_args()

    # load cfg yaml
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    # apply fallback values
    cfg = apply_args_to_cfg(cfg, args)

    # seg classes
    seg_classes = {"standing_trees": [0], "fallen_trees": [1, 2, 3]}

    main_training_loop(cfg, seg_classes)


if __name__ == "__main__":
    main()

