import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def _path(data_dir, name, s_max, suffix=''):
    for S in range(s_max, 6):
        path = f'{data_dir}/{name}_smax{S}{suffix}.pt'
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'{data_dir}/{name}_smax{s_max}{suffix}.pt')


def split(y, seed, train_size=0.5, val_size=0.25):
    idx = np.arange(len(y))
    tv = train_size + val_size
    tv_idx, test_idx, tv_y, _ = train_test_split(idx, y, test_size=1 - tv, stratify=y, random_state=seed)
    train_idx, val_idx, _, _ = train_test_split(tv_idx, tv_y, test_size=val_size / tv, stratify=tv_y, random_state=seed)
    masks = []
    for ids in (train_idx, val_idx, test_idx):
        m = torch.zeros(len(y), dtype=torch.bool)
        m[ids] = True
        masks.append(m)
    return masks


def load(name, task, data_dir, s_max, seed, train_size=0.5, val_size=0.25, x_dir=None):
    path = _path(data_dir, name, s_max, '' if task == 'node' else f'_split{seed}')
    data = torch.load(path, map_location='cpu', weights_only=False)
    keys = [f's{s}' for s in range(1, s_max + 1)]
    data.dist_mats = {k: data.dist_mats[k] for k in keys}
    data.norm_mats = {k: data.norm_mats[k] for k in keys}

    if task == 'node':
        y = data.y.numpy()
        data.train_mask, data.val_mask, data.test_mask = split(y, seed, train_size, val_size)
        num_classes = int(y.max()) + 1
    else:
        num_classes = 2
        if x_dir:
            x = torch.load(f'{x_dir}/{name}_x.pt', map_location='cpu', weights_only=False)['x']
            assert x.shape[0] == data.x.shape[0]
            data.x = x

    return data, data.x.shape[1], num_classes
