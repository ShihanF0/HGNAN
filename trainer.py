import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def step(model, data, y, loss_fn, opt):
    out = model(data).squeeze(-1)[data.train_mask]
    loss = loss_fn(out, y[data.train_mask])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.item()


def metrics(out, y, multi, level='full'):
    r = dict.fromkeys(('auroc', 'auprc', 'precision', 'recall', 'f1'), float('nan'))
    pred = torch.softmax(out, dim=-1).argmax(-1) if multi else (torch.sigmoid(out) > 0.5).long()
    r['acc'] = (pred == y).sum().item() / y.numel()
    if level == 'light':
        return r
    out, y = out.cpu(), y.cpu().numpy()
    prob = (torch.softmax(out, dim=-1) if multi else torch.sigmoid(out)).numpy()
    try:
        r['auroc'] = roc_auc_score(y, prob, multi_class='ovr') if multi else roc_auc_score(y, prob)
    except ValueError:
        pass
    if level == 'auroc':
        return r
    pred = prob.argmax(-1) if multi else (prob > 0.5).astype(int)
    avg = 'macro' if multi else 'binary'
    try:
        r['auprc'] = average_precision_score(y, prob, average='macro') if multi else average_precision_score(y, prob)
    except ValueError:
        pass
    r['precision'] = precision_score(y, pred, average=avg, zero_division=0)
    r['recall'] = recall_score(y, pred, average=avg, zero_division=0)
    r['f1'] = f1_score(y, pred, average=avg, zero_division=0)
    return r


@torch.no_grad()
def evaluate(model, data, y, mask, loss_fn, multi, level='full'):
    model.eval()
    out = model(data).squeeze(-1)[mask]
    r = metrics(out, y[mask], multi, level)
    r['loss'] = loss_fn(out, y[mask]).item()
    return r
