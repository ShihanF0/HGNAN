import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def mlp(d_in, d_hid, d_out, n_layers, bias, dropout=None):
    if n_layers == 1:
        return [nn.Linear(d_in, d_out, bias=bias)]
    layers = []
    for i in range(n_layers - 1):
        layers += [nn.Linear(d_in if i == 0 else d_hid, d_hid, bias=bias), nn.ReLU()]
        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(d_hid, d_out, bias=bias))
    return layers


class HGNAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels, bias=True, dropout=0.0,
                 device='cuda', normalize=True, weight=True, num_s=1, aggregation='overall',
                 attn_hidden=64, attn_dropout=0.2, lam=0.015, budget=1 << 31):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.dropout = dropout
        self.normalize = normalize
        self.weight = weight
        self.aggregation = aggregation
        self.budget = budget
        self._shell = {}
        self._edges = {}

        if weight:
            self.feature_weights = nn.Parameter(torch.rand(in_channels))
        self.s_weights = nn.Parameter(torch.ones(num_s))

        if aggregation == 'neighbor':
            self.lam = lam
            self.attn_dropout = attn_dropout
            self.mix_logit = nn.Parameter(torch.tensor(2.0))
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.bias_same = nn.Parameter(torch.tensor(0.0))
            self.bias_near = nn.Parameter(torch.tensor(0.0))
            self.attn_mlp = nn.Sequential(
                nn.Linear(4 * out_channels, attn_hidden),
                nn.ELU(),
                nn.Dropout(p=attn_dropout),
                nn.Linear(attn_hidden, 1),
            )
            with torch.no_grad():
                nn.init.zeros_(self.attn_mlp[-1].weight)
                nn.init.zeros_(self.attn_mlp[-1].bias)

        self.fs = nn.ModuleList(
            nn.Sequential(*mlp(1, hidden_channels, out_channels, num_layers, bias, dropout))
            for _ in range(in_channels))
        self.m = nn.Sequential(*mlp(1, hidden_channels, 1, num_layers, bias))

    def _fs(self, x, lo, hi):
        h = x[:, lo:hi].t().unsqueeze(-1)
        for i, layer in enumerate(self.fs[0]):
            if isinstance(layer, nn.Linear):
                W = torch.stack([self.fs[k][i].weight for k in range(lo, hi)]).transpose(1, 2)
                if layer.bias is not None:
                    b = torch.stack([self.fs[k][i].bias for k in range(lo, hi)]).unsqueeze(1)
                    h = torch.baddbmm(b, h, W)
                else:
                    h = torch.bmm(h, W)
            elif isinstance(layer, nn.ReLU):
                h = torch.relu(h)
            elif isinstance(layer, nn.Dropout):
                h = F.dropout(h, p=layer.p, training=self.training)
        return h

    def _chunk(self, x):
        per = 4 * x.size(0) * (self.hidden_channels or self.out_channels) * max(self.num_layers, 1)
        return max(1, min(x.size(1), int(self.budget / max(per, 1))))

    def phi(self, x):
        w = F.softmax(self.feature_weights, dim=0) if self.weight else None
        p = x.size(1)
        chunk = self._chunk(x)
        ckpt = self.training and torch.is_grad_enabled() and chunk < p

        def block(lo, hi):
            fx = self._fs(x, lo, hi)
            return (fx * w[lo:hi].view(-1, 1, 1)).sum(0) if w is not None else fx.sum(0)

        out = None
        for a in range(0, p, chunk):
            b = min(p, a + chunk)
            s = cp.checkpoint(block, a, b, use_reentrant=False) if ckpt else block(a, b)
            out = s if out is None else out + s
        return out

    def _shells(self, key, dist, norm):
        if key not in self._shell:
            uniq, idx = torch.unique(dist.detach().to('cpu'), return_inverse=True)
            self._shell[key] = (uniq.view(-1, 1).to(self.device), idx.to(self.device),
                                norm.to(self.device) if self.normalize else None)
        return self._shell[key]

    def overall(self, h, dist, norm, key):
        uniq, idx, norm = self._shells(key, dist, norm)
        W = self.m(uniq).squeeze(-1)[idx]
        if norm is not None:
            W = W / norm
        return W @ h

    def _nbrs(self, key, dist):
        if key not in self._edges:
            d = dist.detach().to('cpu')
            hops = torch.where(d > 0, 1.0 / d.clamp(min=1e-12) - 1.0, torch.full_like(d, float('inf'))).round()
            nb = (hops >= 1) & (hops <= 2)
            nb.fill_diagonal_(False)
            e = nb.nonzero(as_tuple=False)
            self._edges[key] = None if e.numel() == 0 else (
                e[:, 0].to(self.device), e[:, 1].to(self.device), (hops == 1)[e[:, 0], e[:, 1]].to(self.device))
        return self._edges[key]

    def neighbor(self, h, dist, key):
        edges = self._nbrs(key, dist)
        if edges is None:
            return h
        i, j, same = edges
        N = h.size(0)

        hi, hj = h[i], h[j]
        logits = self.attn_mlp(torch.cat([hi, hj, (hi - hj).abs(), hi * hj], dim=1)).squeeze(-1)
        logits = logits + torch.where(same, self.bias_same, self.bias_near)
        e = self.leaky_relu(logits)

        sum_e = torch.zeros(N, device=h.device).index_add(0, i, e)
        cnt = torch.zeros(N, device=h.device).index_add(0, i, torch.ones_like(e))
        e = e - (sum_e / cnt.clamp_min(1.0)).index_select(0, i)
        exp_e = torch.exp(e.clamp(max=10.0))
        denom = torch.zeros(N, device=h.device)
        denom.index_add_(0, i, exp_e)
        alpha = exp_e / (denom.index_select(0, i) + 1e-12)
        if self.attn_dropout > 0.0 and self.training:
            alpha = F.dropout(alpha, p=self.attn_dropout, training=True)

        alpha = F.softshrink(alpha, lambd=self.lam).clamp_min(0.0)
        rs = torch.zeros(N, device=h.device).index_add(0, i, alpha)
        ok = rs.index_select(0, i) > 0
        a_ij = torch.zeros_like(alpha)
        a_ij[ok] = alpha[ok] / (rs.index_select(0, i)[ok] + 1e-12)

        agg = torch.zeros_like(h)
        agg.index_add_(0, i, a_ij.unsqueeze(1) * h[j])
        a = torch.sigmoid(self.mix_logit)
        return a * h + (1.0 - a) * agg

    def agg(self, h, data):
        outs = []
        for key in sorted(data.dist_mats.keys(), key=lambda k: int(k[1:])):
            if self.aggregation == 'overall':
                outs.append(self.overall(h, data.dist_mats[key], data.norm_mats[key], key))
            else:
                outs.append(self.neighbor(h, data.dist_mats[key], key))
        beta = F.softmax(self.s_weights, dim=0).view(-1, 1, 1)
        return (torch.stack(outs, dim=0) * beta).sum(0)

    def forward(self, data):
        return self.agg(self.phi(data.x.to(self.device)), data)


ABLATIONS = ('none', 'no_agg', 'no_additive', 'no_features')


class Ablation(HGNAN):
    def __init__(self, *args, ablation='none', **kwargs):
        if ablation == 'no_features':
            kwargs['aggregation'] = 'overall'
        super().__init__(*args, **kwargs)
        self.ablation = ablation
        if ablation == 'no_additive':
            self.encoder = nn.Sequential(*mlp(
                self.in_channels, self.hidden_channels or self.out_channels, self.out_channels,
                self.num_layers, self.bias, self.dropout)).to(self.device)
            del self.fs
        elif ablation == 'no_features':
            self.const = nn.Parameter(torch.randn(1, self.out_channels))
            del self.fs

    def phi(self, x):
        if self.ablation == 'no_additive':
            return self.encoder(x)
        if self.ablation == 'no_features':
            return self.const.expand(x.size(0), -1).to(self.device)
        return super().phi(x)

    def forward(self, data):
        h = self.phi(data.x.to(self.device))
        return h if self.ablation == 'no_agg' else self.agg(h, data)
