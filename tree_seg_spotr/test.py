import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from openpoints.models import build_model_from_cfg
from openpoints.utils.config import EasyConfig

from tree_seg_spotr.spotr_train_dataloader import TreePartNormalDatasetLargeNumPoints

seg_classes = {"standing_trees": [0], "fallen_trees": [1, 2, 3]}


def build_model_input_from_batch(
    points,
    label,
    target,
    device,
):
    B, N, C = points.shape

    # Split pos and extra features
    pos_np = points[:, :, :3].cpu().numpy()

    pos = torch.from_numpy(pos_np).float().to(device)

    feats = points[:, :, 3:].to(device)

    x = torch.cat([pos, feats], dim=2)

    data = {
        "pos": pos.contiguous(),
        "x": x.permute(0, 2, 1).contiguous(),
        "cls": label,
    }

    return data


def save_pred(combined_arr, out_path):
    dir_name = os.path.dirname(out_path)

    # If no directory is in path, use current directory
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    np.save(out_path, combined_arr)


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

    # print("SEg lablel to cat", seg_label_to_cat)

    accuracy_combination = 0

    # real_targets = np.zeros((1, len()))

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), disable=True)
        for batch_id, (points, label, target, data_chosen, filename, *rest) in pbar:
            points = points.float().to(device)
            label = label.long().to(device)
            target = target.long().to(device)

            # print(points[0].shape)
            # exit()

            B, N, C = points.shape

            data = build_model_input_from_batch(points, label, target, device)

            # Returns Batch, num_classes, points
            logits = model(data)

            cur_pred_val_logits = (
                logits.cpu().data.numpy()
            )  # [Batch, Parts, No of points]

            cur_pred_val = np.zeros((B, N)).astype(np.int32)
            target_np = target.cpu().data.numpy()

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

                # print(sub_logits)
                # exit()

                # print("Logits shape", logits_i.shape)
                # print("Sub Logits shape", sub_logits.shape)

                sub_preds = np.argmax(sub_logits, axis=1)
                # print("Parts", parts)

                cur_pred_val[i, :] = sub_preds + parts[0]

                # print("cur_pred_val", cur_pred_val[i, :])

                # print("Sub Preds. Parts", sub_preds)
                # # print("After Sub Preds. Parts", sub_preds + parts[0])
                # print("TArget ", target_np)
                # exit()

                # target_to_pred[batch_id+i][1] = sub_preds + parts[0]
                # target_to_pred[batch_id+i][1] = sub_preds + parts[0]

                print(
                    "pred",
                    np.unique(cur_pred_val[i, :]),
                    "real",
                    np.unique(target_np[i, :]),
                )

                # if parts[0] == 1:
                #     pred_1_correct.append(np.sum(cur_pred_val[i, :] == target_np[i, :]))

                #     print(np.unique(cur_pred_val[i, :]))
                #     print(np.unique(target_np[i, :]))

            target_transf = (
                torch.from_numpy(target_np).unsqueeze(-1).to(device)
            )  # [1, 2500, 1]
            pred_transf = (
                torch.from_numpy(cur_pred_val).unsqueeze(-1).to(device)
            )  # [1, 2500, 1]

            combined_res = (
                torch.cat([points, target_transf, pred_transf], dim=2)
                .cpu()
                .data.numpy()
            )  # [1, 2500, 9]

            pred_op_path = filename[0].replace("data", "pred")

            save_pred(combined_res, pred_op_path)

            # print(type(combined_res))

            # print(combined_res[0])
            # # print(cur_pred_val.shape)
            # # print(target_np.shape)
            # exit()

            correct = np.sum(cur_pred_val == target_np)
            total_correct += correct
            total_seen += B * N

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

    # pred_op_path = filename[0].replace("data", "pred")

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


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    model = build_model_from_cfg(cfg.model)
    checkpoint = torch.load(
        cfg.pretrained_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    TEST_DATASET = TreePartNormalDatasetLargeNumPoints(
        root=cfg.dataset.train_root,
        split="test",
        use_rgbi=True,
        is_test=True,
    )

    test_loader = DataLoader(
        TEST_DATASET,
        batch_size=1,
        shuffle=False,
        num_workers=5,
    )

    res = evaluate_openpoints(
        model, test_loader, device, seg_classes, cfg.num_part, cfg.num_classes
    )

    print(res)
