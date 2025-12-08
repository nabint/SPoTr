# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
from copy import deepcopy
import torch

warnings.filterwarnings("ignore")


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def rgb_normalize(points):
    min_vals = points.min(axis=0)  # shape: (,3)
    max_vals = points.max(axis=0)  # shape: (,3)

    normalized = (points - min_vals) / (max_vals - min_vals + 1e-8)  # avoid div by zero

    return normalized


def intensity_normalize(points):
    min_vals = points.min()  # shape: (,3)
    max_vals = points.max()  # shape: (,3)

    normalized = (points - min_vals) / (max_vals - min_vals + 1e-8)  # avoid div by zero

    return normalized


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    #     device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #     device = xyz.device
    xyz = xyz[:, :, :3]
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class TreePartNormalDatasetLargeNumPoints(Dataset):
    def __init__(
        self,
        root="./data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
        npoints=2500,
        split="train",
        class_choice=None,
        normal_channel=False,
        use_rgbi=False,
        is_test=False,
    ):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, "category.txt")
        self.cat = {}
        self.normal_channel = normal_channel
        self.use_rgbi = use_rgbi
        self.is_test = is_test

        print("use_rgbi: ", self.use_rgbi)

        with open(self.catfile, "r") as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(
            os.path.join(
                self.root, "train_test_split", "shuffled_train_file_list.json"
            ),
            "r",
        ) as f:
            train_ids_orig = set([str(d.split("/")[-1]) for d in json.load(f)])
        #         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
        #             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(
            os.path.join(self.root, "train_test_split", "shuffled_test_file_list.json"),
            "r",
        ) as f:
            test_ids_orig = set([str(d.split("/")[-1]) for d in json.load(f)])

        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], "numpy")
            fns = sorted(os.listdir(dir_point))

            # filter point cloud that has less than 2048 points
            train_ids = []
            test_ids = []

            for fn in fns:
                if fn in train_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    train_ids.append(fn)
                #                     if pc_data.shape[0] >= 2048:
                #                         train_ids.append(fn)

                elif fn in test_ids_orig:
                    fn_path = os.path.join(dir_point, fn)
                    pc_data = np.load(fn_path)
                    test_ids.append(fn)
            #                     if pc_data.shape[0] >= 2048:
            #                         test_ids.append(fn)

            if split == "trainval":
                fns = [fn for fn in fns if (fn in train_ids)]
            #                 print("train: ", fns)
            elif split == "train":
                fns = [fn for fn in fns if fn in train_ids]
            #             elif split == 'val':
            #                 fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == "test":
                fns = [fn for fn in fns if fn in test_ids]
            #                 print("test: ", fns)
            else:
                print("Unknown split: %s. Exiting.." % (split))
                exit(-1)

            # print(os.path.basename(fns))
            #             print(fns)
            for fn in fns:
                #                 token = (os.path.splitext(os.path.basename(fn))[0])
                #                 print("token: ", token)
                self.meta[item].append(os.path.join(dir_point, fn))

        #         print("cat: ", self.cat)
        self.datapath = []
        for item in self.cat:
            #             print(item)
            #             print(self.meta[item])
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        print("classes: ", self.classes)

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {"standing_trees": [0], "fallen_trees": [1, 2, 3]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    #         print("datapath: ", self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg, filename = self.cache[index]
        else:
            fn = self.datapath[index]
            filename = fn[1]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            #             data = np.loadtxt(fn[1]).astype(np.float32)
            data = np.load(fn[1]).astype(np.float32)

            #             if not self.normal_channel:
            #                 point_set = deepcopy(data[:, 0:3])
            #             else:
            #                 point_set = deepcopy(data[:, 0:6])

            if not self.use_rgbi:
                point_set = deepcopy(data[:, 0:3])
            else:
                point_set = deepcopy(data[:, 0:7])  # x,y,z,r,g,b,intensity
            #             point_set = deepcopy(data)

            seg = data[:, -1].astype(
                np.int32
            )  # for all point clouds last index is for parts
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg, fn[1])

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # N, C

        if self.use_rgbi:
            point_set[:, 3:6] = rgb_normalize(point_set[:, 3:6])
            point_set[:, 6] = intensity_normalize(point_set[:, 6])

        point_set_all_pts = deepcopy(point_set)
        seg_all_pts = deepcopy(seg)

        point_set_exp = np.expand_dims(point_set, axis=0)
        fps_idx = farthest_point_sample(
            torch.tensor(point_set_exp), npoint=self.npoints
        )
        new_data = index_points(point_set_exp, fps_idx)

        seg_new = np.expand_dims(seg, axis=0)  # 1, N
        seg_new = np.expand_dims(seg_new, axis=-1)  # 1, N, 1
        seg_new = torch.tensor(seg_new)
        seg_new = index_points(seg_new, fps_idx)
        seg = seg_new[0, :, :]
        seg = np.array(seg)
        seg = seg.reshape(-1)
        #         print(seg[0, 0])

        #         any_greater = np.any(seg > 3)
        #         if any_greater:
        #             print(np.unique(seg))

        point_set = new_data[0, :, :]
        #         print(point_set.shape, seg.shape)

        if self.is_test:
            return (
                point_set,
                cls,
                seg,
                point_set,
                filename,
                point_set_all_pts,
                seg_all_pts,
            )

        return point_set, cls, seg, point_set, filename

    def __len__(self):
        return len(self.datapath)
