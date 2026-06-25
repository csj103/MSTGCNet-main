import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


def cv_squared(x, eps=1e-10):
    if x.numel() <= 1:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    return x.float().var(unbiased=False) / (x.float().mean().pow(2) + eps)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TrendSeasonalRouter(nn.Module):
    def __init__(self, d_model, num_experts, trend_kernel_sizes, seasonality_k, noisy):
        super().__init__()
        self.num_experts = num_experts
        self.trend_kernel_sizes = trend_kernel_sizes
        self.seasonality_k = seasonality_k
        self.noisy = noisy
        self.trend_score = nn.Linear(d_model, len(trend_kernel_sizes))
        self.merge = nn.Linear(d_model, d_model)
        self.router = nn.Linear(d_model, num_experts)
        self.noise = nn.Linear(d_model, num_experts)

    def _seasonal(self, x):
        # x: [B, L, D]
        freq = torch.fft.rfft(x, dim=1)
        amp = freq.abs().mean(dim=-1)
        amp[:, 0] = 0
        k = min(self.seasonality_k, amp.size(1))
        top_idx = torch.topk(amp, k, dim=1).indices
        mask = torch.zeros_like(freq)
        gather = top_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
        mask.scatter_(1, gather, 1.0 + 0.0j)
        return torch.fft.irfft(freq * mask, n=x.size(1), dim=1)

    def _trend(self, x):
        # Multi-kernel average pooling with sample-adaptive weights.
        pooled = []
        xt = x.transpose(1, 2)
        for kernel_size in self.trend_kernel_sizes:
            pad_left = (kernel_size - 1) // 2
            pad_right = kernel_size - 1 - pad_left
            trend = F.avg_pool1d(
                F.pad(xt, (pad_left, pad_right), mode="replicate"),
                kernel_size=kernel_size,
                stride=1,
            ).transpose(1, 2)
            pooled.append(trend)
        stacked = torch.stack(pooled, dim=-1)
        weights = torch.softmax(self.trend_score(x.mean(dim=1)), dim=-1)
        return torch.einsum("bldc,bc->bld", stacked, weights)

    def forward(self, x):
        seasonal = self._seasonal(x)
        trend = self._trend(x)
        transformed = self.merge(x + seasonal + trend)
        pooled = transformed.mean(dim=1)
        logits = self.router(pooled)
        if self.training and self.noisy:
            noise_scale = F.softplus(self.noise(pooled)) + 1e-2
            logits = logits + torch.randn_like(logits) * noise_scale
        weights = torch.softmax(logits, dim=-1)
        return weights, transformed


class CCSTGCNExpert(nn.Module):
    def __init__(
        self,
        seq_len,
        num_vars,
        d_model,
        patch_size,
        knn_k,
        attn_heads,
        dropout,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_vars = num_vars
        self.patch_size = patch_size
        self.num_patches = math.ceil(seq_len / patch_size)
        self.num_nodes = num_vars * self.num_patches
        self.knn_k = min(knn_k, self.num_nodes)

        self.to_vars = nn.Linear(d_model, num_vars)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=1, num_heads=1, dropout=dropout, batch_first=True
        )
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, patch_size))
        self.graph_weight = nn.Parameter(torch.empty(patch_size, patch_size))
        self.graph_bias = nn.Parameter(torch.zeros(patch_size))
        self.to_model = nn.Linear(num_vars, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.graph_weight)

        patch_ids = torch.arange(self.num_patches).repeat_interleave(num_vars)
        causal = patch_ids.unsqueeze(1) >= patch_ids.unsqueeze(0)
        self.register_buffer("causal_mask", causal.float())

    def _patch(self, x):
        # x: [B, L, D] -> [B, nodes, patch_size]
        vars_x = self.to_vars(x).transpose(1, 2)
        pad_len = self.num_patches * self.patch_size - self.seq_len
        if pad_len > 0:
            vars_x = F.pad(vars_x, (0, pad_len))
        patches = vars_x.reshape(
            x.size(0), self.num_vars, self.num_patches, self.patch_size
        )
        return patches.permute(0, 2, 1, 3).reshape(
            x.size(0), self.num_nodes, self.patch_size
        )

    def _unpatch(self, node_features):
        bsz = node_features.size(0)
        patches = node_features.reshape(
            bsz, self.num_patches, self.num_vars, self.patch_size
        ).permute(0, 2, 1, 3)
        vars_x = patches.reshape(bsz, self.num_vars, self.num_patches * self.patch_size)
        vars_x = vars_x[:, :, : self.seq_len].transpose(1, 2)
        return self.to_model(vars_x)

    def _adjacency(self):
        emb = F.normalize(self.node_embeddings, dim=-1)
        sim = torch.relu(torch.matmul(emb, emb.transpose(0, 1)))
        sim = sim * self.causal_mask
        if self.knn_k < self.num_nodes:
            top_idx = torch.topk(sim, self.knn_k, dim=-1).indices
            mask = torch.zeros_like(sim)
            mask.scatter_(1, top_idx, 1.0)
            sim = sim * mask
        adj = sim + torch.eye(self.num_nodes, device=sim.device, dtype=sim.dtype)
        degree = adj.sum(dim=-1).clamp_min(1e-6)
        norm = degree.rsqrt().unsqueeze(1) * adj * degree.rsqrt().unsqueeze(0)
        return norm

    def forward(self, x):
        patches = self._patch(x)
        attn_in = patches.reshape(-1, self.patch_size, 1)
        attn_out, _ = self.temporal_attn(attn_in, attn_in, attn_in)
        patches = attn_out.reshape(x.size(0), self.num_nodes, self.patch_size)

        adj = self._adjacency()
        graph_out = torch.einsum("ij,bjf->bif", adj, patches)
        graph_out = torch.matmul(graph_out, self.graph_weight) + self.graph_bias
        graph_out = F.gelu(graph_out)
        graph_out = self.dropout(graph_out)
        return self._unpatch(graph_out), adj


class GMoEBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        num_vars,
        d_model,
        d_ff,
        num_experts,
        patch_sizes,
        top_k,
        knn_k,
        attn_heads,
        dropout,
        trend_kernel_sizes,
        seasonality_k,
        noisy_gating,
        abl_gcn=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.abl_gcn = abl_gcn
        if len(patch_sizes) < num_experts:
            patch_sizes = list(patch_sizes) + [patch_sizes[-1]] * (
                num_experts - len(patch_sizes)
            )

        self.router = TrendSeasonalRouter(
            d_model=d_model,
            num_experts=num_experts,
            trend_kernel_sizes=trend_kernel_sizes,
            seasonality_k=seasonality_k,
            noisy=noisy_gating,
        )
        self.experts = nn.ModuleList(
            [
                CCSTGCNExpert(
                    seq_len=seq_len,
                    num_vars=num_vars,
                    d_model=d_model,
                    patch_size=patch_sizes[i],
                    knn_k=knn_k,
                    attn_heads=attn_heads,
                    dropout=dropout,
                )
                for i in range(num_experts)
            ]
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        weights, transformed = self.router(x)
        top_values, top_indices = torch.topk(weights, self.top_k, dim=-1)
        gates = torch.zeros_like(weights)
        gates.scatter_(1, top_indices, top_values)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        expert_outputs = []
        adj_list = []
        for expert in self.experts:
            out, adj = expert(transformed)
            expert_outputs.append(out)
            adj_list.append(adj)

        stacked = torch.stack(expert_outputs, dim=-1)
        mixed = torch.einsum("blde,be->bld", stacked, gates)
        x = self.norm1(x + mixed)
        x = self.norm2(x + self.ffn(x))

        importance = gates.sum(dim=0)
        load = (gates > 0).float().sum(dim=0)
        balance_loss = cv_squared(importance) + cv_squared(load)
        return x, balance_loss, adj_list


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.revin = bool(configs.revin)
        self.loss_coef = configs.loss_coef

        self.conv_embedding = nn.Conv1d(
            in_channels=configs.enc_in,
            out_channels=configs.d_model,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        self.conv_scale = nn.Parameter(torch.ones(1))
        self.position_embedding = PositionalEncoding(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        patch_size_list = configs.patch_size_list
        if patch_size_list and isinstance(patch_size_list[0], int):
            patch_size_list = [patch_size_list for _ in range(configs.e_layers)]

        self.blocks = nn.ModuleList()
        for layer_idx in range(configs.e_layers):
            self.blocks.append(
                GMoEBlock(
                    seq_len=configs.seq_len,
                    num_vars=configs.enc_in,
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    num_experts=configs.num_experts_list[layer_idx],
                    patch_sizes=patch_size_list[layer_idx],
                    top_k=configs.top_k,
                    knn_k=configs.knn_k,
                    attn_heads=configs.attn_heads,
                    dropout=configs.dropout,
                    trend_kernel_sizes=configs.trend_kernel_sizes,
                    seasonality_k=configs.seasonality_k,
                    noisy_gating=configs.noisy_gating == 1,
                    abl_gcn=bool(configs.abl_GCN),
                )
            )

        self.reconstruction = nn.Linear(configs.d_model, configs.enc_in)
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark=None, x_dec=None, x_mark_dec=None):
        if self.revin:
            x = self.revin_layer(x_enc, "norm")
        else:
            x = x_enc

        emb = self.conv_embedding(x.transpose(1, 2)).transpose(1, 2)
        emb = self.conv_scale * emb + self.position_embedding(emb)
        emb = self.dropout(emb)

        balance_loss = torch.zeros((), dtype=x_enc.dtype, device=x_enc.device)
        for block in self.blocks:
            emb, aux_loss, _ = block(emb)
            balance_loss = balance_loss + aux_loss

        dec_out = self.reconstruction(emb)
        if self.revin:
            dec_out = self.revin_layer(dec_out, "denorm")
        return dec_out, self.loss_coef * balance_loss
