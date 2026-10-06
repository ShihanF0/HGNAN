import argparse
import copy
import csv
import os
import time

import numpy as np
import torch

from datasets import load
from models import ABLATIONS, HGNAN, Ablation
from trainer import evaluate, step


class EarlyStop:
    def __init__(self, patience, minimize=True):
        self.patience = patience
        self.minimize = minimize
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, score):
        if self.minimize:
            score = -score
        if self.best is None:
            self.best = score
        elif score < self.best + 1e-5:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best = score
            self.counter = 0


def run(args, seed, idx, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()

    data, num_features, num_classes = load(args.data_name, args.task, args.data_dir, args.s_max, seed,
                                           args.train_size, args.val_size, args.x_dir)
    multi = num_classes > 2
    loss_fn = torch.nn.CrossEntropyLoss() if multi else torch.nn.BCEWithLogitsLoss()
    y = (data.y.long() if multi else data.y.float()).flatten().to(device)
    data.x = data.x.to(device)
    for k in ('train_mask', 'val_mask', 'test_mask'):
        setattr(data, k, getattr(data, k).to(device))

    kw = dict(in_channels=num_features, out_channels=num_classes if multi else 1, num_layers=args.n_layers,
              hidden_channels=args.hidden_channels, dropout=args.dropout, device=device,
              weight=not args.no_weights, aggregation=args.aggregation, num_s=args.s_max)
    model = (HGNAN(**kw) if args.ablation == 'none' else Ablation(ablation=args.ablation, **kw)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    by_auroc = args.select == 'val_auroc'
    stopper = EarlyStop(args.patience, minimize=not by_auroc)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max' if by_auroc else 'min',
                                                       factor=0.95, patience=8, min_lr=1e-7)
    best, best_state = -float('inf') if by_auroc else float('inf'), None

    for epoch in range(args.epochs):
        model.train(epoch == 0 or args.dropout_all)
        loss = step(model, data, y, loss_fn, opt)
        val = evaluate(model, data, y, data.val_mask, loss_fn, multi, 'auroc' if by_auroc else 'light')
        score = val['auroc'] if by_auroc else val['loss']
        sched.step(score)
        if (score > best) if by_auroc else (score < best):
            best, best_state = score, copy.deepcopy(model.state_dict())
        if epoch % args.log_every == 0:
            print(f'run {idx} epoch {epoch:4d}  loss {loss:.4f}  val loss {val["loss"]:.4f}  val acc {val["acc"]:.4f}'
                  + (f'  val auroc {val["auroc"]:.4f}' if by_auroc else ''), flush=True)
        stopper(score)
        if stopper.stop or loss < args.loss_thresh:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if args.save_dir:
            out = os.path.join(args.save_dir, args.data_name)
            os.makedirs(out, exist_ok=True)
            torch.save({'state_dict': {k: v.cpu() for k, v in best_state.items()}, 'args': vars(args), 'seed': seed},
                       os.path.join(out, f'run{idx}_seed{seed}.pt'))

    val = evaluate(model, data, y, data.val_mask, loss_fn, multi)
    test = evaluate(model, data, y, data.test_mask, loss_fn, multi)
    print(f'run {idx} (seed {seed})  test acc {test["acc"]:.4f}  auroc {test["auroc"]:.4f}  '
          f'auprc {test["auprc"]:.4f}  f1 {test["f1"]:.4f}', flush=True)
    del model, opt
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return dict(run=idx, seed=seed, val_acc=val['acc'], val_auroc=val['auroc'], test_acc=test['acc'],
                test_auroc=test['auroc'], test_auprc=test['auprc'], test_f1=test['f1'], duration_s=time.time() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['node', 'edge'], required=True)
    p.add_argument('--data_name', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--x_dir', default=None)
    p.add_argument('--aggregation', choices=['overall', 'neighbor'], default='overall')
    p.add_argument('--s_max', type=int, default=1)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--hidden_channels', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--dropout_all', action='store_true')
    p.add_argument('--no_weights', action='store_true')
    p.add_argument('--ablation', choices=ABLATIONS, default='none')
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--select', choices=['val_loss', 'val_auroc'], default='val_loss')
    p.add_argument('--loss_thresh', type=float, default=1e-4)
    p.add_argument('--runs', type=int, default=10)
    p.add_argument('--train_size', type=float, default=0.5)
    p.add_argument('--val_size', type=float, default=0.25)
    p.add_argument('--results_csv', default='')
    p.add_argument('--save_dir', default='')
    p.add_argument('--log_every', type=int, default=50)
    args = p.parse_args()
    print(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    np.random.seed(0)
    seeds = np.random.randint(low=0, high=10000, size=args.runs).tolist()
    res = [run(args, int(s), i, device) for i, s in enumerate(seeds)]

    def ms(k):
        v = np.array([r[k] for r in res], dtype=np.float64) * 100
        return f'{v.mean():.2f} ± {v.std():.2f}'
    print(f'Val Acc:    {ms("val_acc")}\nVal AUROC:  {ms("val_auroc")}\n'
          f'Test Acc:   {ms("test_acc")}\nTest AUROC: {ms("test_auroc")}\nTest AUPRC: {ms("test_auprc")}')

    if args.results_csv:
        os.makedirs(os.path.dirname(args.results_csv) or '.', exist_ok=True)
        with open(args.results_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(res[0]))
            w.writeheader()
            w.writerows(res)


if __name__ == '__main__':
    main()
