import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        out = torch.einsum("ij,...jd->...id", adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

    def forward(self, q, k, v, adj=None):
        attn_mask = None
        if adj is not None:
            attn_mask = adj <= 0
        out, _ = self.attn(q, k, v, attn_mask=attn_mask)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x):
        return self.net(x)


class GraphConvLayer(nn.Module):
    def __init__(
        self, in_dim, out_dim, top_k, knn_k, num_nodes, attn_heads=4, patch_size=2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.knn_k = min(knn_k, num_nodes)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, in_dim))
        self.gcn = GCNLayer(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def generate_temporal_causal_adjacency(self, batch_size, seq_len, num_nodes, device):
        adj = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return adj.unsqueeze(0).repeat(batch_size, 1, 1)

    def adaptive_adjacency(self):
        emb = F.normalize(self.node_embeddings, dim=-1)
        sim = torch.relu(torch.matmul(emb, emb.transpose(0, 1)))
        if self.knn_k < self.num_nodes:
            topk = torch.topk(sim, self.knn_k, dim=-1).indices
            mask = torch.zeros_like(sim)
            mask.scatter_(1, topk, 1.0)
            sim = sim.masked_fill(mask == 0, float("-inf"))
        return torch.softmax(sim, dim=-1)

    def forward(self, x, node_embeddings=None, temporal_embeddings=None):
        adj = self.adaptive_adjacency()
        return self.norm(self.gcn(x, adj))


class GraphBlock(nn.Module):
    def __init__(self, in_dim, out_dim, top_k, knn_k, num_nodes, attn_heads=4):
        super().__init__()
        self.graph = GraphConvLayer(in_dim, out_dim, top_k, knn_k, num_nodes, attn_heads)
        self.ffn = FeedForward(out_dim, out_dim * 4)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, node_embeddings=None, temporal_embeddings=None):
        x = self.graph(x, node_embeddings, temporal_embeddings)
        return self.norm(x + self.ffn(x))


def nconv(x, A):
    return torch.einsum("...nc,nm->...mc", x, A)


def linear(x, adj, num_nodes, in_dim, out_dim):
    layer = GCNLayer(in_dim, out_dim).to(x.device)
    return layer(x, adj)


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
