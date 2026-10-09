import copy

import pytest
import torch

from models.MSTGCNet import Model
from run import build_parser, build_setting, normalize_args


def arguments(*options):
    return normalize_args(build_parser().parse_args(["--use_gpu", "false", *options]))


def test_named_ablation_matches_legacy_switch_and_isolates_results():
    baseline = arguments()
    named = arguments("--without_graph")
    legacy = arguments("--abl_GCN", "1")
    assert named.abl_GCN == legacy.abl_GCN == 1
    assert vars(named) == vars(legacy)
    differences = {k for k in vars(baseline) if getattr(baseline, k) != getattr(named, k)}
    assert differences == {"abl_GCN"}
    assert "_no_graph_cfg" in build_setting(named, 0)
    assert build_setting(named, 0) == build_setting(legacy, 0)
    assert build_setting(baseline, 0) != build_setting(named, 0)


def test_graph_switch_rejects_invalid_values():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--abl_GCN", "2"])


def test_ablation_ignores_graph_but_trains_attention_and_ffn():
    torch.manual_seed(2025)
    model = Model(arguments("--abl_GCN", "1")).eval()
    expert = model.blocks[0].experts[0]

    def forbidden_graph():
        raise AssertionError("Graph construction must not execute")

    for block in model.blocks:
        for item in block.experts:
            item._adjacency = forbidden_graph
    inputs = torch.randn(2, 96, 64)
    output, _ = expert(inputs)
    changed = copy.deepcopy(expert)
    with torch.no_grad():
        changed.graph_weight.fill_(1000)
        changed.graph_bias.fill_(-1000)
        changed.node_embeddings.zero_()
        assert torch.equal(output, changed(inputs)[0])
    target = torch.randn_like(output)
    (output - target).square().mean().backward()
    assert expert.graph_weight.grad is None
    assert expert.graph_bias.grad is None
    for parameter in (expert.temporal_attn.in_proj_weight, expert.ffn[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.train()
    batch = torch.randn(2, 96, 10)
    reconstruction, balance = model(batch)
    loss = (reconstruction - batch).square().flatten(1).sum(1).mean() + balance
    loss.backward()
    assert reconstruction.shape == batch.shape
    assert torch.isfinite(loss)
    assert model.blocks[0].router.router.weight.grad is not None
