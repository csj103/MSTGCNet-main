import torch

from models.MSTGCNet import Model
from run import build_parser, normalize_args


def build_model():
    args = normalize_args(build_parser().parse_args(["--use_gpu", "false"]))
    return Model(args)


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


def test_router_keeps_sample_specific_temporal_information():
    torch.manual_seed(2025)
    model = build_model().eval()
    inputs = torch.randn(64, 96, 10)
    with torch.no_grad():
        hidden = model.revin_layer(inputs, "norm")
        hidden = model.conv_embedding(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = model.conv_scale * hidden + model.position_embedding(hidden)
        gates, _, _ = model.blocks[0].router(hidden)

    assert gates.std(dim=0).mean() > 1e-3
    assert torch.allclose(gates.sum(dim=1), torch.ones(64), atol=1e-6)
    assert torch.all((gates > 0).sum(dim=1) == model.blocks[0].top_k)
