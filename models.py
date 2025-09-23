import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F
import numpy as np
from utils import *

class HGNAN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers,
        hidden_channels=None,
        bias=True,
        dropout=0.0,
        device='cuda',
        limited_m=True,
        normalize_m=True,
        m_per_feature=False,
        weight = True,
        num_s_values=1,
        aggregation = "overall",

        edge_temp=1.0, # GAT temp
        attn_dropout=0.2, # attention dropout
        use_soft_threshold=True, # soft-shrink or not
        soft_threshold_lambda=0.015, # soft-shrink threshold
        lambda_l1=1e-4, # L1 coefficient
    ):
        
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.dropout = dropout
        self.limited_m = limited_m
        self.normalize_m = normalize_m
        self.m_per_feature = m_per_feature
        self.weight = weight
        self.aggregation = aggregation
        if self.weight == True:
            self.feature_weights = nn.Parameter(torch.rand(self.in_channels))
        # Create weight for different s_values
        self.s_weights = nn.Parameter(torch.ones(num_s_values))

        if self.aggregation == "neighbor":
            self.use_soft_threshold = use_soft_threshold
            self.soft_threshold_lambda = soft_threshold_lambda
            self.lambda_l1 = lambda_l1
            self.mix_logit = nn.Parameter(torch.tensor(2.0))

            self.edge_temp = edge_temp
            self.attn_dropout = attn_dropout
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.gamma_logit = nn.Parameter(torch.tensor(-2.0, device=self.device))
            self.reg_l1_alpha = torch.tensor(0.0, device=self.device)
            self.bias_same = nn.Parameter(torch.tensor(0.0))
            self.bias_near = nn.Parameter(torch.tensor(0.0))
            
            self.attn_mlp = nn.Sequential(
                nn.Linear(4 * self.out_channels, getattr(self, "attn_hidden", 64), bias=True),
                nn.ELU(),
                nn.Dropout(p=self.attn_dropout),
                nn.Linear(getattr(self, "attn_hidden", 64), 1, bias=True)
            )
            with torch.no_grad():
                nn.init.zeros_(self.attn_mlp[-1].weight)
                nn.init.zeros_(self.attn_mlp[-1].bias)

        # shape functions f_k
        self.fs = nn.ModuleList()
        for _ in range(in_channels):
            if num_layers == 1:
                layers = [nn.Linear(1, out_channels, bias=bias)]
            else:
                layers = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                for _ in range(1, num_layers - 1):
                    layers += [nn.Linear(hidden_channels, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.fs.append(nn.Sequential(*layers))

        # distance functions \rho
        if m_per_feature:
            self.ms = nn.ModuleList()
            for _ in range(out_channels if limited_m else in_channels):
                if num_layers == 1:
                    m_layers = [nn.Linear(1, out_channels, bias=bias)]
                else:
                    m_layers = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                    for _ in range(1, num_layers - 1):
                        m_layers += [nn.Linear(hidden_channels, hidden_channels, bias=bias), nn.ReLU()]
                    if limited_m:
                        m_layers.append(nn.Linear(hidden_channels, 1, bias=bias))
                    else:
                        m_layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))
                self.ms.append(nn.Sequential(*m_layers))
        else:
            if num_layers == 1:
                m_layers = [nn.Linear(1, out_channels, bias=bias)]
            else:
                m_layers = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                for _ in range(1, num_layers - 1):
                    m_layers += [nn.Linear(hidden_channels, hidden_channels, bias=bias), nn.ReLU()]
                if limited_m:
                    m_layers.append(nn.Linear(hidden_channels, 1, bias=bias))
                else:
                    m_layers.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.m = nn.Sequential(*m_layers)

    def _calculate_aggregation(self, f_sums, distances, normalization_matrix):
        if self.aggregation == "overall":
            m_dist = self.m(distances.flatten().view(-1, 1))
            m_dist = m_dist.view(distances.size(0), distances.size(1), self.out_channels)

            if self.normalize_m:
                m_dist = m_dist / normalization_matrix.unsqueeze(-1)

            output = torch.sum(m_dist * f_sums.unsqueeze(0), dim=1)

        elif self.aggregation == "neighbor":
            N, D = f_sums.size()
            h = f_sums                                         # (N, D)

            neighbor_mask = torch.isin(distances, torch.tensor([1.0, 0.5], device=distances.device))
            neighbor_mask.fill_diagonal_(False)
            edges = neighbor_mask.nonzero(as_tuple=False)      # (E, 2)
            if edges.numel() == 0:
                return h

            i = edges[:, 0]  # target indices (E,)
            j = edges[:, 1]  # source indices (E,)

            # GAT-style weight
            hi, hj = h[i], h[j]
            pair = torch.cat([hi, hj, (hi - hj).abs(), (hi * hj)], dim=1)

            # logits
            logits = self.attn_mlp(pair).squeeze(-1) 
            is_same = (distances[i, j] == 1.0)
            logits = logits + torch.where(is_same, self.bias_same, self.bias_near)
            e = self.leaky_relu((logits) / max(self.edge_temp, 1e-12))

            # softmax within each neighbors
            sum_e = torch.zeros(N, device=h.device).index_add(0, i, e)
            cnt_e = torch.zeros(N, device=h.device).index_add(0, i, torch.ones_like(e))
            mean_e = sum_e / cnt_e.clamp_min(1.0)
            e_shift = e - mean_e.index_select(0, i)

            exp_e = torch.exp(e_shift.clamp(max=10.0))
            denom = torch.zeros(N, device=h.device)
            denom.index_add_(0, i, exp_e)
            alpha = exp_e / (denom.index_select(0, i) + 1e-12) # (E,)

            if self.attn_dropout > 0.0 and self.training:
                alpha = F.dropout(alpha, p=self.attn_dropout, training=True)

            # soft-threshold
            if self.use_soft_threshold and self.soft_threshold_lambda > 0:
                alpha_shrunk = F.softshrink(alpha, lambd=self.soft_threshold_lambda).clamp_min(0.0)
            else:
                alpha_shrunk = alpha

            sum_per_row = torch.zeros(N, device=h.device).index_add(0, i, alpha_shrunk)
            deg_row = torch.zeros(N, device=h.device).index_add(0, i, torch.ones_like(alpha_shrunk))
            l1_per_row = (sum_per_row / deg_row.clamp_min(1.0)).mean()
            self.reg_l1_alpha = self.lambda_l1 * l1_per_row

            denom2 = sum_per_row
            safe_edge = denom2.index_select(0, i) > 0
            alpha_final = torch.zeros_like(alpha_shrunk)
            alpha_final[safe_edge] = alpha_shrunk[safe_edge] / (denom2.index_select(0, i)[safe_edge] + 1e-12)

            agg = torch.zeros_like(h)
            agg.index_add_(0, i, alpha_final.unsqueeze(1) * h[j])  # (N, D)

            a = torch.sigmoid(self.mix_logit)
            output = a * h + (1.0 - a) * agg

            with torch.no_grad():
                avg_neighbors = neighbor_mask.sum(dim=1).float().mean()
                kept_per_row = torch.zeros(N, device=h.device).index_add(0, i, (alpha_final > 0).float())
                self.avg_neighbors_no_cutoff = float(avg_neighbors.item())
                self.avg_neighbors_after_cutoff = float((kept_per_row.mean()).item())
                print(f"[HGNAN] avg neighbors (no cutoff): {self.avg_neighbors_no_cutoff:.3f} | "
                      f"avg neighbors after shrink: {self.avg_neighbors_after_cutoff:.3f} ")
        else:
            raise ValueError("Unknown aggregation type: {}".format(self.aggregation))
        return output
    
    def forward(self, inputs):
        x = inputs.x.to(self.device)
        fx = torch.empty(x.size(0), x.size(1), self.out_channels).to(self.device)
        for feature_index in range(x.size(1)):
            feature_col = x[:, feature_index].view(-1, 1)
            fx[:, feature_index] = self.fs[feature_index](feature_col)
        if self.weight == True:
            attention_weights = F.softmax(torch.exp(self.feature_weights), dim=0)
            fx_weighted = fx * attention_weights.unsqueeze(0).unsqueeze(-1)  # (N, num_features, out_channels)
            f_sums = fx_weighted.sum(dim=1)
        else:
            f_sums = fx.sum(dim=1)

        # Loop to deal with all matrixs of s_values
        s_outputs = []
        sorted_s_keys = sorted(inputs.dist_mats.keys(), key=lambda k: int(k[1:]))

        for s_key in sorted_s_keys:
            distances = inputs.dist_mats[s_key].to(self.device)
            norm_mat = inputs.norm_mats[s_key].to(self.device)
            s_output = self._calculate_aggregation(f_sums, distances, norm_mat)
            s_outputs.append(s_output)
        
        s_weights_normalized = F.softmax(self.s_weights, dim=0)
        stacked_outputs = torch.stack(s_outputs, dim=0)
        weights_reshaped = s_weights_normalized.view(-1, 1, 1)
        final_output = torch.sum(stacked_outputs * weights_reshaped, dim=0)

        return final_output

    def print_m_params(self):
        if hasattr(self, 'm'):
            print("Single m network parameters:")
            for name, param in self.m.named_parameters():
                print(name, param)
        elif hasattr(self, 'ms'):
            print("Separate m networks per dimension:")
            for idx, module in enumerate(self.ms):
                for name, param in module.named_parameters():
                    print(f"ms[{idx}].{name}", param)
        else:
            print("No m parameters found.")