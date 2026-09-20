import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.GraphGCNLayer import GraphConvLayer


def cv_squared(x, eps=1e-10):
    if x.numel() <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x.float().var(unbiased=False) / (x.float().mean().pow(2) + eps)


def noisy_top_k_gating(logits, noise_epsilon=1e-2, top_k=2):
    noisy_logits = logits
    if noise_epsilon > 0:
        noisy_logits = logits + torch.randn_like(logits) * noise_epsilon
    top_k = min(top_k, logits.size(-1))
    values, indices = torch.topk(noisy_logits, top_k, dim=-1)
    gates = torch.zeros_like(logits)
    gates.scatter_(-1, indices, torch.softmax(values, dim=-1))
    return gates


class GMoE(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        device=None,
        k=2,
        num_nodes=10,
        patch_size=None,
        noisy_gating=True,
        d_model=64,
        residual_connection=True,
        attn_heads=2,
        dropout=0.1,
        trend_kernel_sizes=None,
        seasonality_k=3,
        knn_k=5,
        abl_GCN=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.residual_connection = residual_connection
        self.abl_GCN = abl_GCN

        if patch_size is None:
            patch_size = [3, 5, 7, 9]
        if isinstance(patch_size, int):
            patch_size = [patch_size]
        if len(patch_size) < num_experts:
            patch_size = list(patch_size) + [patch_size[-1]] * (
                num_experts - len(patch_size)
            )

        self.experts = nn.ModuleList()
        for kernel_size in patch_size[:num_experts]:
            padding = kernel_size // 2
            self.experts.append(
                nn.Sequential(
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=d_model,
                    ),
                    nn.GELU(),
                    nn.Conv1d(d_model, d_model, kernel_size=1),
                    nn.Dropout(dropout),
                )
            )

        self.gate = nn.Linear(d_model, num_experts)
        self.graph = GraphConvLayer(d_model, d_model, k, knn_k, num_nodes, attn_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, loss_coef=1e-2, node_embeddings=None, temporal_embeddings=None):
        batch, seq_len, d_model = x.shape
        logits = self.gate(x.mean(dim=1))
        gates = noisy_top_k_gating(
            logits,
            noise_epsilon=1e-2 if self.training and self.noisy_gating else 0.0,
            top_k=self.k,
        )

        expert_input = x.transpose(1, 2)
        outputs = []
        for expert in self.experts:
            out = expert(expert_input)
            if out.size(-1) != seq_len:
                out = out[..., :seq_len]
            outputs.append(out.transpose(1, 2))
        stacked = torch.stack(outputs, dim=-1)
        mixed = torch.einsum("btde,be->btd", stacked, gates)

        importance = gates.sum(dim=0)
        load = (gates > 0).float().sum(dim=0)
        balance_loss = loss_coef * (cv_squared(importance) + cv_squared(load))

        if self.residual_connection:
            mixed = mixed + x
        mixed = self.norm(mixed)

        adj_list = []
        if not self.abl_GCN:
            adj_list.append(self.graph.adaptive_adjacency())
        return mixed, balance_loss, adj_list


class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
        self.modes1 = modes1

    def forward(self, x):
        freq = torch.fft.rfft(x, dim=1)
        freq[:, self.modes1 :] = 0
        filtered = torch.fft.irfft(freq, n=x.size(1), dim=1)
        return self.proj(filtered)


def series_decomp_multi(x, kernel_size):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]
    trends = []
    for kernel in kernel_size:
        padding = kernel // 2
        trend = F.avg_pool1d(
            x.transpose(1, 2), kernel_size=kernel, stride=1, padding=padding
        ).transpose(1, 2)
        if trend.size(1) != x.size(1):
            trend = trend[:, : x.size(1)]
        trends.append(trend)
    trend = torch.stack(trends, dim=0).mean(dim=0)
    seasonal = x - trend
    return seasonal, trend


class GraphBlock(nn.Module):
    def __init__(self, in_dim, out_dim, top_k, knn_k, num_nodes, attn_heads=4):
        super().__init__()
        self.graph = GraphConvLayer(in_dim, out_dim, top_k, knn_k, num_nodes, attn_heads)

    def forward(self, x, node_embeddings=None, temporal_embeddings=None):
        return self.graph(x, node_embeddings, temporal_embeddings)


def nconv(x, A):
    return torch.einsum("...nc,nm->...mc", x, A)


def linear(x, adj, num_nodes, in_dim, out_dim):
    layer = nn.Linear(in_dim, out_dim).to(x.device)
    return nconv(layer(x), adj)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout):
        super().__init__()
        self.gdep = gdep
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Linear((gdep + 1) * c_in, c_out)

    def forward(self, x, adj, mask=None):
        outputs = [x]
        h = x
        for _ in range(self.gdep):
            h = nconv(h, adj)
            outputs.append(h)
        return self.dropout(self.mlp(torch.cat(outputs, dim=-1)))
