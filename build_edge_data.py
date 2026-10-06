import argparse
import math
import os

import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

INF = float('inf')


def neg_sample(S):
    S = torch.tensor(S, dtype=torch.float32)
    neg = torch.zeros_like(S)
    for e in range(S.shape[1]):
        col = S[:, e]
        mem = torch.where(col != 0)[0]
        if len(mem) <= 1:
            continue
        n_rm = math.floor(len(mem) * 0.5)
        perm = torch.randperm(len(mem))
        rm, keep = mem[perm[:n_rm]], mem[perm[n_rm:]]
        rest = torch.where(col == 0)[0]
        if len(rest) < len(rm):
            continue
        new = rest[torch.randperm(len(rest))[:len(rm)]]
        neg[keep, e] = col[keep]
        neg[new, e] = col[rm]
    return torch.cat((S, neg), dim=1), neg.shape[1]


def dist(inc, anchor, s, device):
    inc = inc.to(device).float()
    M = inc.shape[1]
    A = inc[:, anchor]
    T = A.shape[1]
    inf = torch.tensor(INF, device=device)

    adj = ((A.T @ A) >= s).float()
    adj.fill_diagonal_(0)
    dT = torch.as_tensor(dijkstra(csr_matrix(adj.cpu().numpy()), directed=False, unweighted=True),
                         dtype=torch.float32, device=device)
    B = (inc.T @ A) >= s

    F = torch.full((M, T), INF, device=device)
    for a in range(0, M, 32):
        b = min(M, a + 32)
        F[a:b] = torch.where(B[a:b].unsqueeze(-1), dT.unsqueeze(0), inf).amin(1)
    d = torch.full((M, M), INF, device=device)
    for a in range(0, M, 8):
        b = min(M, a + 8)
        d[:, a:b] = (torch.where(B[a:b].unsqueeze(1), F.unsqueeze(0), inf).amin(-1) + 2.0).T

    direct = (inc.T @ inc) >= s
    d = torch.where(direct, torch.minimum(d, torch.ones_like(d)), d)
    d.fill_diagonal_(0.0)
    return torch.minimum(d, d.T).cpu()


def shells(d):
    uniq, inv = torch.unique(d, return_inverse=True)
    inv = inv.view(d.shape)
    cnt = torch.zeros(d.shape[0], uniq.numel(), dtype=torch.float32)
    cnt.scatter_add_(1, inv, torch.ones_like(inv, dtype=torch.float32))
    return cnt.gather(1, inv)


def rich_x(X, S):
    F = X.shape[1]
    E = S.shape[1]
    mem = (S != 0).to(X.dtype)
    out = torch.zeros(E, 3 * F + 3, dtype=X.dtype)
    out[:, :F] = torch.mm(S.T, X)
    for e in range(E):
        idx = torch.nonzero(mem[:, e], as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        Z = X[idx]
        out[e, F:2 * F] = Z.mean(0)
        out[e, 2 * F:3 * F] = Z.std(0, unbiased=False) if Z.shape[0] > 1 else 0.0
        if Z.shape[0] > 1:
            inter = Z @ Z.T
            c = Z.sum(1, keepdim=True)
            tan = inter / (c + c.T - inter).clamp_min(1e-12)
            iu = torch.triu_indices(Z.shape[0], Z.shape[0], offset=1)
            pw = tan[iu[0], iu[1]]
            out[e, 3 * F] = pw.mean()
            out[e, 3 * F + 1] = pw.std(unbiased=False) if pw.numel() > 1 else 0.0
        out[e, 3 * F + 2] = float(idx.numel())
    return out


def build(name, s_max, args, device):
    X = torch.load(f'{args.raw_dir}/{name}/{name}.pt', map_location='cpu')
    S = loadmat(f'{args.raw_dir}/{name}/{name}.mat')[name]['S']
    if isinstance(S, np.ndarray) and S.shape == (1, 1):
        S = S[0, 0]

    torch.manual_seed(args.neg_seed)
    np.random.seed(args.neg_seed)
    H, n_neg = neg_sample(S)
    n_pos = S.shape[1]
    M = H.shape[1]
    inc = (H != 0).float()
    x = torch.mm(H.T, X).float()
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    yt = torch.as_tensor(y, dtype=torch.float)
    print(f'{name}: {inc.shape[0]} metabolites, {n_pos} reactions')

    if args.rich_x_dir:
        os.makedirs(args.rich_x_dir, exist_ok=True)
        torch.save({'x': rich_x(X.float(), H)}, f'{args.rich_x_dir}/{name}_x.pt')

    tv = args.train_size + args.val_size
    for seed in args.split_seeds:
        idx = np.arange(M)
        tv_idx, test_idx, tv_y, _ = train_test_split(idx, y, test_size=1 - args.train_size - args.val_size,
                                                     stratify=y, random_state=seed)
        train_idx, val_idx = train_test_split(tv_idx, test_size=args.val_size / tv, stratify=tv_y,
                                              random_state=seed)[:2]
        masks = {}
        for k, ids in (('train', train_idx), ('val', val_idx), ('test', test_idx)):
            masks[k] = torch.zeros(M, dtype=torch.bool)
            masks[k][ids] = True
        anchor = masks['train'] & (yt == 1)

        dmats, nmats = {}, {}
        for s in range(1, s_max + 1):
            d = dist(inc, anchor, s, device)
            nmats[f's{s}'] = shells(d)
            dmats[f's{s}'] = 1.0 / (d + 1.0)

        data = Data(x=x, h=H, y=yt, dist_mats=dmats, norm_mats=nmats,
                    train_mask=masks['train'], val_mask=masks['val'], test_mask=masks['test'])
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(data, f'{args.out_dir}/{name}_smax{s_max}_split{seed}.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['iAF1260b', 'iJR904', 'iSB619', 'iYO844'])
    ap.add_argument('--s_max', type=int, default=2)
    ap.add_argument('--raw_dir', default='data/raw')
    ap.add_argument('--out_dir', default='data/processed/edge')
    ap.add_argument('--rich_x_dir', default='')
    ap.add_argument('--train_size', type=float, default=0.5)
    ap.add_argument('--val_size', type=float, default=0.25)
    ap.add_argument('--neg_seed', type=int, default=42)
    ap.add_argument('--split_seeds', type=int, nargs='+', default=None)
    args = ap.parse_args()
    if args.split_seeds is None:
        np.random.seed(0)
        args.split_seeds = np.random.randint(0, 10000, 10).tolist()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for ds in args.datasets:
        build(ds, args.s_max, args, device)


if __name__ == '__main__':
    main()
