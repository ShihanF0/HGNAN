import argparse
import os
import pickle

import numpy as np
import torch
from scipy.sparse import issparse
from scipy.sparse.csgraph import dijkstra
from torch_geometric.data import Data


def read_allset(name, raw_dir):
    table = np.genfromtxt(os.path.join(raw_dir, name, f'{name}.content'), dtype=np.dtype(str))
    x = torch.from_numpy(table[:, 1:-1].astype(np.float32))
    y = torch.LongTensor(table[:, -1].astype(float))
    if name == 'zoo':
        y = y - 1
    ids = {j: i for i, j in enumerate(np.array(table[:, 0], dtype=np.int32))}
    raw = np.genfromtxt(os.path.join(raw_dir, name, f'{name}.edges'), dtype=np.int32)
    edges = np.array(list(map(ids.get, raw.flatten())), dtype=np.int32).reshape(raw.shape)
    n = edges[:, 0].max() + 1
    H = torch.zeros(n, edges[:, 1].max() - n + 1)
    H[edges[:, 0], edges[:, 1] - n] = 1.0
    return x[:n], y[:n], H


def read_pickle(name, raw_dir):
    def rd(f):
        with open(os.path.join(raw_dir, name, f), 'rb') as fh:
            return pickle.load(fh)
    feat = rd('features.pickle')
    feat = np.asarray(feat.todense() if issparse(feat) else feat)
    y = np.asarray(rd('labels.pickle')).reshape(-1)
    hg = rd('hypergraph.pickle')
    try:
        eids = sorted(hg.keys(), key=int)
    except (TypeError, ValueError):
        eids = sorted(hg.keys())
    H = torch.zeros(len(y), len(eids))
    for j, e in enumerate(eids):
        for v in set(hg[e]):
            H[int(v), j] = 1.0
    return torch.as_tensor(feat, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long), H


def edge_dist(H, s):
    adj = (H.t() @ H) >= s
    adj.fill_diagonal_(True)
    D = dijkstra(adj.numpy().astype(np.int8), directed=False, unweighted=True)
    return torch.from_numpy(D).float()


def segmin(rows, idx, src_rows, src, width):
    out = torch.full((rows, width), float('inf'))
    chunk = max(1, int(2 ** 30 / (4 * max(width, 1))))
    for a in range(0, idx.numel(), chunk):
        b = min(idx.numel(), a + chunk)
        out.index_reduce_(0, idx[a:b], src[src_rows[a:b]], 'amin', include_self=True)
    return out


def node_dist(H, s):
    N, E = H.shape
    D = edge_dist(H, s)
    nz = (H != 0).nonzero(as_tuple=False)
    rows, cols = nz[:, 0].contiguous(), nz[:, 1].contiguous()
    M = segmin(N, rows, cols, D, E)
    d = segmin(N, rows, cols, M.t().contiguous(), N) + 1.0
    d.fill_diagonal_(0.0)
    return d


def shells(d):
    N = d.shape[0]
    fin = torch.isfinite(d)
    K = int(d[fin].max().item()) + 1 if fin.any() else 0
    out = torch.empty_like(d)
    step = max(1, int(2 ** 30 / (8 * max(N, 1))))
    for a in range(0, N, step):
        blk = d[a:a + step]
        codes = torch.where(torch.isfinite(blk), blk.round().long(), torch.full_like(blk, K, dtype=torch.long))
        cnt = torch.zeros(blk.shape[0], K + 1)
        cnt.scatter_add_(1, codes, torch.ones_like(blk))
        out[a:a + step] = cnt.gather(1, codes)
    return out


def build(name, s_max, raw_dir, out_dir):
    if name in ('zoo', 'Mushroom', 'NTU2012'):
        x, y, H = read_allset(name, raw_dir)
    else:
        x, y, H = read_pickle(name, raw_dir)
    N, E = H.shape
    print(f'{name}: {N} nodes, {E} hyperedges, {x.shape[1]} features')

    dist, norm = {}, {}
    for s in range(1, s_max + 1):
        d = node_dist(H, s)
        dist[f's{s}'] = torch.where(torch.isfinite(d), 1.0 / (1.0 + d), torch.zeros_like(d))
        norm[f's{s}'] = shells(d)

    data = Data(x=x, y=y, h=H, dist_mats=dist, norm_mats=norm)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(data, f'{out_dir}/{name}_smax{s_max}.pt')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--s_max', type=int, default=1)
    ap.add_argument('--raw_dir', default='data/raw')
    ap.add_argument('--out_dir', default='data/processed/node')
    a = ap.parse_args()
    for ds in a.datasets:
        build(ds, a.s_max, a.raw_dir, a.out_dir)
