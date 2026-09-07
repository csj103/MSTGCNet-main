import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN


GRAPH_TRACE_POSITIONS = [
    "input",
    "after_adjacency",
    "residual_base",
    "after_residual",
    "after_norm",
]


def cv_squared(values):
    if values.numel() <= 1:
        return torch.zeros((), dtype=values.dtype, device=values.device)
    values = values.float()
    return values.var(unbiased=False) / (values.mean().pow(2) + 1e-10)


class RelativeValueEmbedding(nn.Module):
    def __init__(self, seq_len, num_vars, d_model, dropout):
        super().__init__()
        self.value_projection = nn.Linear(1, d_model)
        self.time_embedding = nn.Parameter(torch.zeros(1, seq_len, 1, d_model))
        self.variable_embedding = nn.Embedding(num_vars, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.variable_embedding.weight, std=0.02)

    def forward(self, x, spectral=None):
        # x: [B, L, N] -> [B, L, N, D]
        bsz, seq_len, num_vars = x.shape
        values = self.value_projection(x.unsqueeze(-1))
        time = self.time_embedding[:, :seq_len]
        var_ids = torch.arange(num_vars, device=x.device)
        variables = self.variable_embedding(var_ids).view(1, 1, num_vars, -1)
        hidden = values + time + variables
        if spectral is not None:
            hidden = hidden + self.value_projection(spectral.unsqueeze(-1))
        return self.dropout(self.norm(hidden))


class SpectralEnergyEnhancer(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.log_threshold_scale = nn.Parameter(torch.zeros(1))
        self.low_gain = nn.Parameter(torch.tensor(0.25))
        self.high_gain = nn.Parameter(torch.tensor(0.0))
        self.temperature = max(float(temperature), 1e-3)

    def forward(self, x):
        # x: [B, L, N]. Low-energy frequency components are softly enhanced.
        freq = torch.fft.rfft(x, dim=1)
        energy = freq.abs().pow(2)
        threshold = energy.mean(dim=1, keepdim=True) * self.log_threshold_scale.exp()
        high_gate = torch.sigmoid((energy - threshold) / self.temperature)
        low_gate = 1.0 - high_gate
        enhanced = freq * (1.0 + self.high_gain * high_gate + self.low_gain * low_gate)
        return torch.fft.irfft(enhanced, n=x.size(1), dim=1)


class TemporalVariableGraphExpert(nn.Module):
    def __init__(
        self,
        seq_len,
        num_vars,
        d_model,
        d_ff,
        patch_size,
        attn_heads,
        knn_k,
        dropout,
        disable_graph=False,
        graph_mode="learned",
        graph_residual_mode="shared",
        graph_residual_alpha=1.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_vars = num_vars
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_patches = math.ceil(seq_len / patch_size)
        self.knn_k = min(knn_k, max(num_vars - 1, 1))
        if graph_mode not in {"learned", "identity", "off"}:
            raise ValueError("graph_mode must be one of: learned, identity, off")
        if disable_graph and graph_mode == "learned":
            graph_mode = "identity"
        if graph_residual_mode not in {"shared", "expert", "mixed"}:
            raise ValueError(
                "graph_residual_mode must be one of: shared, expert, mixed"
            )
        self.graph_mode = graph_mode
        self.graph_residual_mode = graph_residual_mode
        self.graph_residual_alpha = float(graph_residual_alpha)
        self.disable_graph = graph_mode != "learned"
        heads = max(1, math.gcd(attn_heads, d_model))
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.variable_embeddings = nn.Parameter(torch.randn(num_vars, d_model) * 0.02)
        self.norm_graph = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _patch_tokens(self, hidden):
        bsz, seq_len, num_vars, d_model = hidden.shape
        pad_len = self.num_patches * self.patch_size - seq_len
        if pad_len > 0:
            hidden = F.pad(hidden, (0, 0, 0, 0, 0, pad_len))
        hidden = hidden.permute(0, 2, 1, 3).contiguous()
        patches = hidden.view(
            bsz, num_vars, self.num_patches, self.patch_size, d_model
        )
        return patches.mean(dim=3)

    def _adjacency(self):
        if self.graph_mode in {"identity", "off"}:
            return torch.eye(
                self.num_vars,
                device=self.variable_embeddings.device,
                dtype=self.variable_embeddings.dtype,
            )
        emb = F.normalize(self.variable_embeddings, dim=-1)
        sim = torch.matmul(emb, emb.transpose(0, 1))
        sim = sim.masked_fill(
            torch.eye(self.num_vars, device=sim.device, dtype=torch.bool),
            float("-inf"),
        )
        top_idx = torch.topk(sim, self.knn_k, dim=-1).indices
        adj = torch.zeros_like(sim)
        adj.scatter_(1, top_idx, 1.0)
        adj = adj + torch.eye(self.num_vars, device=sim.device, dtype=sim.dtype)
        degree = adj.sum(dim=-1).clamp_min(1e-6)
        return degree.rsqrt().unsqueeze(1) * adj * degree.rsqrt().unsqueeze(0)

    def _expand_tokens(self, tokens, seq_len):
        bsz, num_vars, _, d_model = tokens.shape
        return (
            tokens.unsqueeze(3)
            .expand(bsz, num_vars, self.num_patches, self.patch_size, d_model)
            .reshape(bsz, num_vars, self.num_patches * self.patch_size, d_model)
            [:, :, :seq_len]
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward(self, hidden, collect_graph_states=False):
        bsz, seq_len, num_vars, d_model = hidden.shape
        residual_base = hidden
        tokens = self._patch_tokens(hidden)
        attn_in = tokens.reshape(bsz * num_vars, self.num_patches, d_model)
        attn_out, _ = self.temporal_attn(attn_in, attn_in, attn_in)
        tokens = attn_out.reshape(bsz, num_vars, self.num_patches, d_model)
        pre_graph_hidden = self._expand_tokens(tokens, seq_len)
        adj = self._adjacency()
        if self.graph_mode == "off":
            expanded = pre_graph_hidden
            after_residual = pre_graph_hidden
            hidden = pre_graph_hidden
        else:
            graph_tokens = torch.einsum("ij,bjpd->bipd", adj, tokens)
            expanded = self._expand_tokens(graph_tokens, seq_len)
            if self.graph_residual_mode == "expert":
                residual_source = pre_graph_hidden
            elif self.graph_residual_mode == "mixed":
                residual_source = (
                    self.graph_residual_alpha * residual_base
                    + (1.0 - self.graph_residual_alpha) * pre_graph_hidden
                )
            else:
                residual_source = residual_base
            after_residual = residual_source + self.dropout(expanded)
            hidden = self.norm_graph(after_residual)
        post_graph_hidden = hidden
        hidden = self.norm_ffn(hidden + self.dropout(self.ffn(hidden)))
        if collect_graph_states:
            graph_trace = {
                "input": pre_graph_hidden,
                "after_adjacency": expanded,
                "residual_base": residual_base,
                "after_residual": after_residual,
                "after_norm": post_graph_hidden,
            }
            return hidden, adj, pre_graph_hidden, post_graph_hidden, graph_trace
        return hidden, adj


class TimeSeriesRouter(nn.Module):
    def __init__(self, seq_len, num_experts, top_k, noisy=True, disabled=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noisy = noisy
        self.disabled = disabled
        self.router = nn.Linear(seq_len, num_experts)
        self.noise = nn.Linear(seq_len, num_experts)
        self.softplus = nn.Softplus()
        self.last_expert_load = None
        self.last_entropy = None

    def forward(self, hidden):
        bsz, seq_len, _, _ = hidden.shape
        if self.disabled:
            gates = torch.full(
                (bsz, self.num_experts),
                1.0 / self.num_experts,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            load = torch.full(
                (self.num_experts,),
                float(bsz),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        else:
            routing_input = hidden.mean(dim=(2, 3))
            if routing_input.size(1) != self.router.in_features:
                routing_input = F.interpolate(
                    routing_input.unsqueeze(1),
                    size=self.router.in_features,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            clean_logits = self.router(routing_input)
            if self.training and self.noisy:
                noise_std = self.softplus(self.noise(routing_input)) + 1e-2
                logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            else:
                logits = clean_logits
            top_logits, top_indices = logits.topk(self.top_k, dim=1)
            top_gates = torch.softmax(top_logits, dim=1)
            gates = torch.zeros_like(logits).scatter(1, top_indices, top_gates)
            load = (gates > 0).sum(dim=0).to(hidden.dtype)

        self.last_expert_load = (gates > 0).sum(dim=0).detach()
        self.last_entropy = (
            -(gates * gates.clamp_min(1e-12).log()).sum(dim=1).mean().detach()
        )
        return gates, load


class AdaptiveMultiScaleBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        num_vars,
        d_model,
        d_ff,
        patch_sizes,
        top_k,
        knn_k,
        attn_heads,
        dropout,
        noisy_gating,
        disable_graph=False,
        graph_mode="learned",
        graph_residual_mode="shared",
        graph_residual_alpha=1.0,
        disable_router=False,
    ):
        super().__init__()
        self.num_experts = len(patch_sizes)
        self.top_k = min(top_k, self.num_experts)
        self.router = TimeSeriesRouter(
            seq_len=seq_len,
            num_experts=self.num_experts,
            top_k=self.top_k,
            noisy=noisy_gating,
            disabled=disable_router,
        )
        self.experts = nn.ModuleList(
            [
                TemporalVariableGraphExpert(
                    seq_len=seq_len,
                    num_vars=num_vars,
                    d_model=d_model,
                    d_ff=d_ff,
                    patch_size=patch_size,
                    attn_heads=attn_heads,
                    knn_k=knn_k,
                    dropout=dropout,
                    disable_graph=disable_graph,
                    graph_mode=graph_mode,
                    graph_residual_mode=graph_residual_mode,
                    graph_residual_alpha=graph_residual_alpha,
                )
                for patch_size in patch_sizes
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _pairwise_expert_cosine(expert_outputs):
        flattened = [
            output[:, -1].reshape(output.size(0), -1)
            for output in expert_outputs
        ]
        pairs = []
        for left in range(len(flattened)):
            for right in range(left + 1, len(flattened)):
                pairs.append(
                    F.cosine_similarity(
                        flattened[left],
                        flattened[right],
                        dim=-1,
                    )
                )
        if pairs:
            return torch.stack(pairs, dim=1)
        return expert_outputs[0].new_zeros(expert_outputs[0].size(0), 0)

    def forward(self, hidden, force_expert_index=None, collect_diagnostics=False):
        gates, load = self.router(hidden)
        expert_outputs = []
        pre_graph_outputs = []
        post_graph_outputs = []
        graph_trace_outputs = {
            position: []
            for position in GRAPH_TRACE_POSITIONS
        }
        adjacencies = []
        for expert in self.experts:
            if collect_diagnostics:
                expert_output, adj, pre_graph, post_graph, graph_trace = expert(
                    hidden,
                    collect_graph_states=True,
                )
                pre_graph_outputs.append(pre_graph)
                post_graph_outputs.append(post_graph)
                for position in GRAPH_TRACE_POSITIONS:
                    graph_trace_outputs[position].append(graph_trace[position])
            else:
                expert_output, adj = expert(hidden)
            expert_outputs.append(expert_output)
            adjacencies.append(adj)
        if force_expert_index is not None:
            expert_index = int(force_expert_index)
            if expert_index < 0 or expert_index >= self.num_experts:
                raise IndexError("force_expert_index is outside the expert range")
            gates = torch.zeros_like(gates)
            gates[:, expert_index] = 1.0
            load = torch.zeros_like(load)
            load[expert_index] = hidden.size(0)
        stacked = torch.stack(expert_outputs, dim=1)
        mixed = (stacked * gates.view(gates.size(0), -1, 1, 1, 1)).sum(dim=1)
        importance = gates.sum(dim=0)
        balance_loss = cv_squared(importance) + cv_squared(load)
        diagnostics = None
        if collect_diagnostics:
            graph_trace_pair_cosine = torch.stack(
                [
                    self._pairwise_expert_cosine(graph_trace_outputs[position])
                    for position in GRAPH_TRACE_POSITIONS
                ],
                dim=1,
            )
            diagnostics = {
                "gates": gates.detach(),
                "expert_pair_cosine": self._pairwise_expert_cosine(
                    expert_outputs
                ).detach(),
                "expert_pre_graph_pair_cosine": self._pairwise_expert_cosine(
                    pre_graph_outputs
                ).detach(),
                "expert_post_graph_pair_cosine": self._pairwise_expert_cosine(
                    post_graph_outputs
                ).detach(),
                "graph_trace_positions": list(GRAPH_TRACE_POSITIONS),
                "graph_trace_pair_cosine": graph_trace_pair_cosine.detach(),
                "adjacency": torch.stack(adjacencies, dim=0).detach(),
            }
        return self.output_norm(hidden + mixed), balance_loss, adjacencies, diagnostics


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.latent_dim = getattr(configs, "latent_dim", configs.d_model)
        self.revin = bool(configs.revin)
        self.disable_spectral = bool(getattr(configs, "disable_spectral", False))
        self.disable_dynamic_score = bool(
            getattr(configs, "disable_dynamic_score", False)
        )
        self.disable_probabilistic = bool(
            getattr(configs, "disable_probabilistic", False)
        )
        self.objective = getattr(configs, "dtsgad_objective", "reconstruction")
        self.score_fusion = getattr(configs, "score_fusion", "dual")
        self.obs_topk = max(1, min(getattr(configs, "obs_topk", 3), configs.enc_in))
        self.dynamic_score_weight = getattr(configs, "dynamic_score_weight", 1.0)
        self.loss_coef = getattr(configs, "loss_coef", 1e-2)
        self.graph_mode = getattr(configs, "graph_mode", "learned")
        self.graph_residual_mode = getattr(configs, "graph_residual_mode", "shared")
        self.graph_residual_alpha = getattr(configs, "graph_residual_alpha", 1.0)

        self.spectral = SpectralEnergyEnhancer(
            temperature=getattr(configs, "spectral_temperature", 0.2)
        )
        self.embedding = RelativeValueEmbedding(
            seq_len=configs.seq_len,
            num_vars=configs.enc_in,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )

        patch_size_list = configs.patch_size_list
        if patch_size_list and isinstance(patch_size_list[0], int):
            patch_size_list = [patch_size_list for _ in range(configs.e_layers)]
        self.blocks = nn.ModuleList(
            [
                AdaptiveMultiScaleBlock(
                    seq_len=configs.seq_len,
                    num_vars=configs.enc_in,
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    patch_sizes=patch_size_list[layer_idx],
                    top_k=configs.top_k,
                    knn_k=configs.knn_k,
                    attn_heads=configs.attn_heads,
                    dropout=configs.dropout,
                    noisy_gating=configs.noisy_gating == 1,
                    disable_graph=bool(getattr(configs, "disable_graph", False)),
                    graph_mode=self.graph_mode,
                    graph_residual_mode=self.graph_residual_mode,
                    graph_residual_alpha=self.graph_residual_alpha,
                    disable_router=bool(getattr(configs, "disable_router", False)),
                )
                for layer_idx in range(configs.e_layers)
            ]
        )

        self.reconstruction_mu = nn.Linear(configs.d_model, 1)
        self.reconstruction_logvar = nn.Linear(configs.d_model, 1)
        self.posterior_mu = nn.Linear(configs.d_model, self.latent_dim)
        self.posterior_logvar = nn.Linear(configs.d_model, self.latent_dim)
        self.prior_gru = nn.GRU(self.latent_dim, configs.d_model, batch_first=True)
        self.prior_mu = nn.Linear(configs.d_model, self.latent_dim)
        self.prior_logvar = nn.Linear(configs.d_model, self.latent_dim)
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.enc_in, affine=False)

    @staticmethod
    def _gaussian_kl(q_mu, q_logvar, p_mu, p_logvar):
        q_var = q_logvar.exp()
        p_var = p_logvar.exp().clamp_min(1e-6)
        kl = 0.5 * (
            p_logvar
            - q_logvar
            + (q_var + (q_mu - p_mu).pow(2)) / p_var
            - 1.0
        )
        return kl.sum(dim=-1).clamp_min(0.0)

    def _latent_scores(self, hidden):
        latent_context = hidden.mean(dim=2)
        q_mu = self.posterior_mu(latent_context)
        q_logvar = self.posterior_logvar(latent_context).clamp(min=-6.0, max=4.0)
        if self.training:
            eps = torch.randn_like(q_mu)
            z = q_mu + eps * torch.exp(0.5 * q_logvar)
        else:
            z = q_mu
        previous_z = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
        prior_hidden, _ = self.prior_gru(previous_z)
        p_mu = self.prior_mu(prior_hidden)
        p_logvar = self.prior_logvar(prior_hidden).clamp(min=-6.0, max=4.0)
        dynamic_score = self._gaussian_kl(q_mu, q_logvar, p_mu, p_logvar)
        return dynamic_score, q_mu, q_logvar, p_mu, p_logvar

    def forward(
        self,
        x_enc,
        x_mark=None,
        x_dec=None,
        x_mark_dec=None,
        return_dict=False,
        collect_diagnostics=False,
        force_expert_index=None,
        input_mask=None,
        target_next=None,
    ):
        if self.revin:
            target = self.revin_layer(x_enc, "norm")
        else:
            target = x_enc

        model_input = target
        if input_mask is not None:
            model_input = target * input_mask.to(device=target.device, dtype=target.dtype)

        spectral = None if self.disable_spectral else self.spectral(model_input)
        hidden = self.embedding(model_input, spectral=spectral)

        balance_loss = torch.zeros((), dtype=x_enc.dtype, device=x_enc.device)
        adjacencies = []
        router_diagnostics = []
        for block in self.blocks:
            hidden, aux_loss, block_adjacencies, block_diagnostics = block(
                hidden,
                force_expert_index=force_expert_index,
                collect_diagnostics=collect_diagnostics,
            )
            balance_loss = balance_loss + aux_loss
            adjacencies.append(block_adjacencies)
            if collect_diagnostics:
                router_diagnostics.append(block_diagnostics)

        if self.objective == "next_step_prediction":
            prediction_norm = self.reconstruction_mu(hidden[:, -1]).squeeze(-1)
            prediction = (
                self.revin_layer._denormalize(prediction_norm.unsqueeze(1)).squeeze(1)
                if self.revin
                else prediction_norm
            )
            dynamic_score = torch.zeros(
                prediction_norm.size(0),
                1,
                dtype=prediction_norm.dtype,
                device=prediction_norm.device,
            )
            outputs = {
                "prediction": prediction,
                "prediction_norm": prediction_norm,
                "reconstruction": prediction,
                "reconstruction_norm": prediction_norm,
                "dynamic_score": dynamic_score,
                "balance_loss": balance_loss,
                "adjacencies": adjacencies,
                "router_diagnostics": router_diagnostics,
            }
            if target_next is not None:
                target_next = target_next.to(
                    device=prediction_norm.device,
                    dtype=prediction_norm.dtype,
                )
                if self.revin:
                    target_next_norm = self.revin_layer._normalize(
                        target_next.unsqueeze(1)
                    ).squeeze(1)
                else:
                    target_next_norm = target_next
                prediction_score = (target_next_norm - prediction_norm).pow(2)
                channel_score = torch.topk(
                    prediction_score,
                    k=self.obs_topk,
                    dim=-1,
                ).values.mean(dim=-1)
                outputs.update(
                    {
                        "target_next_norm": target_next_norm,
                        "target_norm": target_next_norm,
                        "observation_score": prediction_score.unsqueeze(1),
                        "prediction_score": prediction_score,
                        "total_score": channel_score.unsqueeze(1),
                    }
                )
            if return_dict:
                return outputs
            return prediction, self.loss_coef * balance_loss

        mu_norm = self.reconstruction_mu(hidden).squeeze(-1)
        logvar = self.reconstruction_logvar(hidden).squeeze(-1).clamp(min=-6.0, max=4.0)
        if self.disable_probabilistic:
            logvar = torch.zeros_like(logvar)

        variance = logvar.exp().clamp_min(1e-6)
        observation_score = 0.5 * (
            (target - mu_norm).pow(2) / variance + logvar + math.log(2.0 * math.pi)
        )
        if self.disable_dynamic_score:
            dynamic_score = torch.zeros_like(observation_score.mean(dim=-1))
            q_mu = q_logvar = p_mu = p_logvar = None
        else:
            dynamic_score, q_mu, q_logvar, p_mu, p_logvar = self._latent_scores(hidden)

        channel_score = torch.topk(
            observation_score, k=self.obs_topk, dim=-1
        ).values.mean(dim=-1)
        if self.score_fusion == "obs":
            total_score = channel_score
        elif self.score_fusion == "dyn":
            total_score = dynamic_score
        else:
            total_score = channel_score + self.dynamic_score_weight * dynamic_score

        reconstruction = (
            self.revin_layer(mu_norm, "denorm") if self.revin else mu_norm
        )
        outputs = {
            "reconstruction": reconstruction,
            "reconstruction_norm": mu_norm,
            "reconstruction_logvar": logvar,
            "target_norm": target,
            "input_norm": model_input,
            "observation_score": observation_score,
            "dynamic_score": dynamic_score,
            "total_score": total_score,
            "balance_loss": balance_loss,
            "posterior_mu": q_mu,
            "posterior_logvar": q_logvar,
            "prior_mu": p_mu,
            "prior_logvar": p_logvar,
            "adjacencies": adjacencies,
            "router_diagnostics": router_diagnostics,
        }
        if return_dict:
            return outputs
        return reconstruction, self.loss_coef * balance_loss
