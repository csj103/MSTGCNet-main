import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


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
    def __init__(
        self,
        seq_len,
        d_model,
        num_experts,
        top_k,
        trend_kernel_sizes,
        seasonality_k,
        noisy,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.trend_kernel_sizes = trend_kernel_sizes
        self.seasonality_k = seasonality_k
        self.noisy = noisy
        self.last_expert_load = None
        self.last_entropy = None
        self.trend_score = nn.Linear(1, len(trend_kernel_sizes))
        self.merge = nn.Linear(d_model, d_model)
        self.channel_projection = nn.Linear(d_model, 1)
        self.router = nn.Linear(seq_len, num_experts)
        self.noise = nn.Linear(seq_len, num_experts)

    def _seasonal(self, x):
        # x: [B, L, D]
        freq = torch.fft.rfft(x, dim=1)
        amp = freq.abs()
        if amp.size(1) <= 1:
            return torch.zeros_like(x)
        amp[:, 0, :] = float("-inf")
        k = min(self.seasonality_k, amp.size(1) - 1)
        top_idx = torch.topk(amp, k, dim=1).indices
        mask = torch.zeros_like(freq)
        mask.scatter_(1, top_idx, 1.0 + 0.0j)
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
        weights = torch.softmax(self.trend_score(x.unsqueeze(-1)), dim=-1)
        return (stacked * weights).sum(dim=-1)

    def forward(self, x):
        seasonal = self._seasonal(x)
        trend = self._trend(x)
        transformed = self.merge(x + seasonal + trend)
        routing_input = self.channel_projection(transformed).squeeze(-1)
        clean_logits = self.router(routing_input)
        if self.training and self.noisy:
            noise_std = F.softplus(self.noise(routing_input)) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            logits = clean_logits

        weights = torch.softmax(logits, dim=1)
        selected_indices = weights.topk(self.top_k, dim=1).indices
        hard_gates = torch.zeros_like(weights).scatter(
            1, selected_indices, 1.0
        )
        sparse_weights = weights * hard_gates
        self.last_expert_load = hard_gates.detach().sum(dim=0)
        self.last_entropy = (
            -(weights * weights.clamp_min(1e-12).log()).sum(dim=1).mean().detach()
        )

        # Eq. (25) uses hard expert counts. The straight-through form preserves
        # those counts in the forward pass while providing router gradients.
        balance_gates = hard_gates + weights - weights.detach()
        return sparse_weights, balance_gates, transformed


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
        abl_gcn=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_vars = num_vars
        self.patch_size = patch_size
        self.num_patches = math.ceil(seq_len / patch_size)
        self.num_nodes = num_vars * self.num_patches
        self.knn_k = min(knn_k, max(self.num_nodes - 1, 1))
        self.abl_gcn = abl_gcn

        self.to_vars = nn.Linear(d_model, num_vars)
        attention_heads = max(1, math.gcd(attn_heads, d_model))
        self.attention_in = nn.Linear(1, d_model)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_out = nn.Linear(d_model, 1)
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, patch_size))
        self.graph_weight = nn.Parameter(torch.empty(patch_size, patch_size))
        self.graph_bias = nn.Parameter(torch.zeros(self.num_nodes, patch_size))
        self.to_model = nn.Linear(num_vars, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.graph_weight)

        patch_ids = torch.arange(self.num_patches).repeat_interleave(num_vars)
        causal = patch_ids.unsqueeze(1) >= patch_ids.unsqueeze(0)
        self.register_buffer("causal_mask", causal)

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
        sim = torch.matmul(emb, emb.transpose(0, 1))
        valid = self.causal_mask.clone()
        valid.fill_diagonal_(False)
        sim = sim.masked_fill(~valid, float("-inf"))
        top_idx = torch.topk(sim, self.knn_k, dim=-1).indices
        adj = torch.zeros_like(sim)
        adj.scatter_(1, top_idx, 1.0)
        adj = adj * valid.to(adj.dtype)
        adj = adj + torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype)
        degree = adj.sum(dim=-1).clamp_min(1e-6)
        norm = degree.rsqrt().unsqueeze(1) * adj * degree.rsqrt().unsqueeze(0)
        return norm

    def _temporal_encode(self, patches):
        batch_size, num_nodes, patch_size = patches.shape
        attn_input = self.attention_in(patches.reshape(-1, patch_size, 1))
        attn_output, _ = self.temporal_attn(attn_input, attn_input, attn_input)
        return self.attention_out(attn_output).reshape(
            batch_size, num_nodes, patch_size
        )

    def forward(self, x):
        patches = self._temporal_encode(self._patch(x))

        if self.abl_gcn:
            identity = torch.eye(
                self.num_nodes, device=patches.device, dtype=patches.dtype
            )
            return self._unpatch(patches), identity

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
            seq_len=seq_len,
            d_model=d_model,
            num_experts=num_experts,
            top_k=self.top_k,
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
                    abl_gcn=abl_gcn,
                )
                for i in range(num_experts)
            ]
        )

    def forward(self, x):
        sparse_weights, balance_gates, _ = self.router(x)

        expert_contributions = []
        adj_list = []
        for expert_idx, expert in enumerate(self.experts):
            selected = sparse_weights[:, expert_idx] > 0
            selected_indices = selected.nonzero(as_tuple=False).squeeze(-1)
            contribution = torch.zeros_like(x)
            if selected_indices.numel() > 0:
                expert_output, adj = expert(x.index_select(0, selected_indices))
                expert_weight = sparse_weights.index_select(
                    0, selected_indices
                )[:, expert_idx].view(-1, 1, 1)
                contribution = contribution.index_copy(
                    0, selected_indices, expert_output * expert_weight
                )
            else:
                adj = None
            expert_contributions.append(contribution)
            adj_list.append(adj)

        mixed = torch.stack(expert_contributions, dim=0).sum(dim=0)

        expert_load = balance_gates.sum(dim=0)
        balance_loss = expert_load.var(unbiased=False) / expert_load.mean().clamp_min(
            1e-10
        )
        return x + mixed, balance_loss, adj_list


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
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.conv_embedding.weight, mode="fan_in", nonlinearity="leaky_relu"
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
