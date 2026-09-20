import json
from pathlib import Path

import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models.MSTGCNet import Model
from run import REPRO_PATCH_SIZE_LIST, build_parser, build_setting, normalize_args


def build_model():
    args = normalize_args(build_parser().parse_args(["--use_gpu", "false"]))
    return Model(args)


def test_default_patch_assignment_is_coarse_to_fine():
    args = normalize_args(build_parser().parse_args(["--use_gpu", "false"]))
    assert args.patch_size_list == REPRO_PATCH_SIZE_LIST


def test_router_and_balance_loss_are_differentiable():
    torch.manual_seed(2025)
    model = build_model().train()
    inputs = torch.randn(32, 96, 10)

    outputs, balance_loss = model(inputs)
    loss = (outputs - inputs).pow(2).mean() + balance_loss
    loss.backward()

    assert outputs.shape == inputs.shape
    assert balance_loss.requires_grad
    for block in model.blocks:
        assert block.router.router.weight.grad is not None
        assert block.router.noise.weight.grad is not None
        assert torch.isfinite(block.router.router.weight.grad).all()
        assert torch.isfinite(block.router.noise.weight.grad).all()
        assert block.router.router.weight.grad.norm() > 0
        assert block.router.noise.weight.grad.norm() > 0


def test_reconstruction_loss_matches_window_squared_error():
    outputs = torch.zeros(2, 96, 10)
    targets = torch.ones_like(outputs)
    loss = Exp_Anomaly_Detection._reconstruction_loss(outputs, targets)
    assert loss.item() == 960.0


def test_router_keeps_sample_specific_temporal_information():
    torch.manual_seed(2025)
    model = build_model().eval()
    inputs = torch.randn(64, 96, 10)
    with torch.no_grad():
        hidden = model.revin_layer(inputs, "norm")
        hidden = model.conv_embedding(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = model.conv_scale * hidden + model.position_embedding(hidden)
        sparse_weights, _, _ = model.blocks[0].router(hidden)

    assert sparse_weights.std(dim=0).mean() > 1e-3
    assert torch.allclose(
        sparse_weights.sum(dim=1),
        torch.ones(inputs.size(0)),
        atol=1e-6,
    )
    assert torch.all(
        (sparse_weights > 0).sum(dim=1) == model.blocks[0].top_k
    )


def test_expert_attention_uses_paper_node_patch_dimensions():
    model = build_model()
    for block in model.blocks:
        for expert in block.experts:
            assert expert.temporal_attn.embed_dim == model.d_model
            inputs = torch.randn(2, 96, 64)
            patches = expert._patch(inputs)
            assert patches.shape == (
                2,
                expert.num_nodes,
                expert.patch_size,
            )


def test_temporal_attention_preserves_node_identity():
    torch.manual_seed(2025)
    expert = build_model().blocks[0].experts[0].eval()
    patches = torch.randn(1, expert.num_nodes, expert.patch_size)
    perturbed = patches.clone()
    perturbed[:, -1, :] += 10
    with torch.no_grad():
        original_output = expert._temporal_encode(patches)
        perturbed_output = expert._temporal_encode(perturbed)
    difference = (perturbed_output - original_output).abs().sum(dim=-1)
    assert torch.allclose(difference[:, :-1], torch.zeros_like(difference[:, :-1]))
    assert torch.all(difference[:, -1] > 0)


def test_sparse_gmoe_executes_only_selected_experts():
    torch.manual_seed(2025)
    model = build_model().eval()
    calls = [0, 0]
    hooks = []
    for block in model.blocks:
        for expert in block.experts:
            def count_call(module, inputs, output):
                calls[0] += 1
            hooks.append(expert.register_forward_hook(count_call))
            original_adjacency = expert._adjacency

            def count_adjacency(original=original_adjacency):
                calls[1] += 1
                return original()

            expert._adjacency = count_adjacency

    with torch.no_grad():
        model(torch.randn(1, 96, 10))
    for hook in hooks:
        hook.remove()

    selected_count = sum(block.top_k for block in model.blocks)
    assert calls == [selected_count, selected_count]


def test_expert_ffn_uses_configured_d_ff():
    model = build_model()
    expert = model.blocks[0].experts[0]
    assert expert.ffn[0].out_features == expert.ffn[3].in_features
    assert expert.ffn[0].out_features == 128

    inputs = torch.randn(4, 96, 64)
    outputs, _ = expert(inputs)
    loss = outputs.pow(2).mean()
    loss.backward()

    assert expert.ffn[0].weight.grad is not None
    assert torch.isfinite(expert.ffn[0].weight.grad).all()
    assert expert.ffn[0].weight.grad.norm() > 0


def test_metadata_ratio_and_config_fingerprint_are_in_setting():
    args = normalize_args(build_parser().parse_args(["--use_gpu", "false"]))
    metadata_path = Path(args.root_path) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    setting = build_setting(args, 0)
    changed_args = normalize_args(
        build_parser().parse_args(
            [
                "--use_gpu",
                "false",
                "--paper_strict",
                "false",
                "--alarm_confirmation",
                "3",
            ]
        )
    )
    changed_setting = build_setting(changed_args, 0)
    assert abs(args.anomaly_ratio - metadata["test_anomaly_ratio"] * 100.0) < 1e-6
    assert len(setting) < 120
    assert "_cfg" in setting
    assert setting == build_setting(args, 0)
    assert setting != changed_setting
