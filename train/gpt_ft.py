#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import traceback

# OpenPoints helpers
from openpoints.models import build_model_from_cfg
from openpoints.loss import build_criterion_from_cfg
from train.spotr_train_dataloader import TreePartNormalDatasetLargeNumPoints


def random_scale_point_cloud_torch(batch_xyz, scale_low=0.8, scale_high=1.25):
    B, N, C = batch_xyz.shape
    scales = (
        torch.rand(B, 1, 1, device=batch_xyz.device) * (scale_high - scale_low)
        + scale_low
    )
    return batch_xyz * scales


def shift_point_cloud_torch(batch_xyz, shift_range=0.1):
    B, N, C = batch_xyz.shape
    shifts = (torch.rand(B, 1, 3, device=batch_xyz.device) - 0.5) * 2 * shift_range
    return batch_xyz + shifts


# ---------------- Model loader -----------------
def load_model(cfg, device):
    model = build_model_from_cfg(cfg.model)
    model.to(device)
    return model


# ---------------- Train for one epoch with robust interface -----------------
def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    num_classes,
    num_part,
    expected_in_channels=7,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for idx, (points, label, target, data_chosen, filename) in pbar:
        optimizer.zero_grad()

        points = points.float().to(device)  # [B, N, C_total]
        label = label.long().to(device)  # [B, 1]
        target = target.long().to(device)  # [B, N]

        # Split XYZ
        # points: [B, N, total_channels] (total_channels >= 3)
        pos = points[:, :, :3].contiguous()  # [B, N, 3]

        # extra features
        feat = (
            points[:, :, 3:].contiguous()
            if points.size(2) > 3
            else torch.zeros(points.size(0), points.size(1), 4, device=points.device)
        )

        # Pad or truncate to ensure exactly 4 extra channels
        if feat.size(2) < 4:
            pad = torch.zeros(
                points.size(0), points.size(1), 4 - feat.size(2), device=points.device
            )
            feat = torch.cat([feat, pad], dim=2)
        elif feat.size(2) > 4:
            feat = feat[:, :, :4]

        # Concatenate XYZ + extra features -> [B, N, 7]
        x = torch.cat([pos, feat], dim=2)

        # Permute to [B, C, N] as expected by Conv1d
        data = {
            "pos": pos,  # still [B, N, 3]
            "cls": label,  # [B, 1]
            "x": x.permute(0, 2, 1).contiguous(),  # [B, 7, N]
        }

        # GPU-safe augmentations (optional)
        scale = torch.rand(pos.size(0), 1, 1, device=pos.device) * 0.45 + 0.8
        pos = pos * scale
        shift = (torch.rand(pos.size(0), 1, 3, device=pos.device) - 0.5) * 0.2
        pos = pos + shift
        data["pos"] = pos

        # Forward pass
        logits = model(data)  # [B, N, num_part] or [B, num_part, N]
        if logits.dim() == 3 and logits.shape[-1] == num_part:
            logits = logits.permute(
                0, 2, 1
            ).contiguous()  # [B, N, num_part] -> [B, num_part, N]

        B, P, N = logits.shape
        logits_flat = logits.permute(0, 2, 1).reshape(-1, P)  # [B*N, num_part]
        target_flat = target.view(-1)

        # Compute loss
        if getattr(criterion, "NAME", None) == "MultiShapeCrossEntropy":
            loss = criterion(logits_flat, target_flat, data["cls"])
        else:
            loss = criterion(logits_flat, target_flat)

        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * B
        preds = logits_flat.max(1)[1]
        total_correct += (preds == target_flat).sum().item()
        total_points += target_flat.numel()

        cur_loss = total_loss / ((idx + 1) * train_loader.batch_size)
        cur_acc = total_correct / float(total_points)
        pbar.set_description(f"Train Loss {cur_loss:.4f} Acc {cur_acc:.4f}")

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_correct / float(total_points)
    return avg_loss, avg_acc


# ---------------- Main -----------------
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="config yaml")
    args = parser.parse_args()

    from openpoints.utils.config import EasyConfig

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    print(f"Loading model from config: {cfg.pretrained_path}")
    model = load_model(cfg, device)

    # Datasets
    TRAIN_DATASET = TreePartNormalDatasetLargeNumPoints(
        root=cfg.dataset.train_root, split="trainval", use_rgbi=True
    )
    train_loader = DataLoader(
        TRAIN_DATASET, batch_size=1, shuffle=True, num_workers=10, drop_last=True
    )

    # Optimizer and criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=cfg.decay_rate,
    )
    criterion = build_criterion_from_cfg(cfg.criterion_args).to(device)

    num_classes = 2
    num_part = 4
    use_rgbi_flag = True

    # Training loop (single epoch example)
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_classes,
            num_part,
            # use_rgbi=use_rgbi_flag,
        )
        print(f"Epoch {epoch} training loss, acc:",  train_loss, train_acc)


if __name__ == "__main__":
    main()
