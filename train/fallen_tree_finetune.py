#!/usr/bin/env python3
import os
import glob
import argparse
import copy
import json
from tqdm import tqdm
from openpoints.loss import build_criterion_from_cfg


from openpoints.utils.config import EasyConfig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ---- openpoints helpers ----
from openpoints.models import build_model_from_cfg
from train.spotr_train_dataloader import TreePartNormalDatasetLargeNumPoints


from train import provider


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def load_model(cfg, device):
    model = build_model_from_cfg(cfg.model)
    model.to(device)
    return model


def train_one_epoch(
    model, train_loader, optimizer, criterion, device, num_classes, num_part
):
    """
    Trains the model for one epoch.

    Args:
        model: segmentation model
        train_loader: PyTorch dataloader for training data
        optimizer: torch optimizer
        criterion: loss function
        device: 'cuda' or 'cpu'
        num_classes: number of object classes
        num_part: number of segmentation labels
    """
    model.train()
    mean_correct = []

    for points, label, target, data_chosen, filename in train_loader:
        optimizer.zero_grad()

        # Move everything to the correct device
        points = points.float().to(device)  # [B, N, C] float32
        label = label.long().to(device)
        target = target.long().to(device)

        # Apply augmentation directly on GPU (optional)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])

        # Pass both to the model if supported
        seg_pred, trans_feat = model(points, None)

        # Flatten for loss computation
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1)

        # Compute accuracy
        pred_choice = seg_pred.max(1)[1]
        correct = pred_choice.eq(target).sum().item()
        batch_acc = correct / float(points.size(0) * points.size(2))
        mean_correct.append(batch_acc)

        # Loss and backward
        loss = criterion(seg_pred, target, trans_feat)
        loss.backward()
        optimizer.step()

    # Return average accuracy
    return float(np.mean(mean_correct))


# Main
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="config yaml")

    args = parser.parse_args()

    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)

    print(f"Loading model from config file: {cfg.pretrained_path}")

    model = load_model(cfg, device)

    TRAIN_DATASET = TreePartNormalDatasetLargeNumPoints(
        root=cfg.dataset.train_root, split="trainval", use_rgbi=True
    )

    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=10,
        shuffle=True,
        num_workers=10,
        drop_last=True,
    )

    TEST_DATASET = TreePartNormalDatasetLargeNumPoints(
        root=cfg.dataset.val_root, split="test", is_test=True, use_rgbi=True
    )

    # print(TRAIN_DATASET.)

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=1, shuffle=False, num_workers=10
    )

    print("The number of training data is: %d" % len(TRAIN_DATASET))
    print("The number of test data is: %d" % len(TEST_DATASET))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=cfg.decay_rate,
    )

    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    num_classes = 2
    num_part = 4

    train_one_epoch(
        model,
        trainDataLoader,
        optimizer,
        criterion,
        device,
        num_classes,
        num_part,
    )


if __name__ == "__main__":
    main()
