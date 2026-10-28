"""Single-factor input-padding controls; no trained checkpoint is reused."""
import copy

import pytest
import torch

from models.MSTGCNet import Model
from run import build_parser, build_setting, normalize_args

torch.set_num_threads(2)


def padding_args(mode, padding='default'):
    return normalize_args(build_parser().parse_args([
        '--experiment_mode', mode, '--input_padding', padding, '--use_gpu', 'false',
    ]))


def input_conv(model):
    return model.conv_embedding if hasattr(model, 'conv_embedding') else model.encoder.temporal


@pytest.mark.parametrize('mode,native,changed', [
    ('E0', 'circular', 'replicate'), ('E1', 'replicate', 'circular'),
])
def test_padding_is_the_only_module_change_and_initialization_is_identical(mode, native, changed):
    torch.manual_seed(2025)
    original = Model(padding_args(mode))
    torch.manual_seed(2025)
    control = Model(padding_args(mode, changed))
    assert input_conv(original).padding_mode == native
    assert input_conv(control).padding_mode == changed
    assert original.state_dict().keys() == control.state_dict().keys()
    assert all(torch.equal(value, control.state_dict()[name])
               for name, value in original.state_dict().items())
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in control.parameters())
    assert repr(original).replace('padding_mode=' + native, 'padding_mode=' + changed, 1) == repr(control)
    # An interior position is unaffected, while the two boundaries change.
    x = torch.arange(96, dtype=torch.float32)[None, None].repeat(2, input_conv(original).in_channels, 1)
    a, b = input_conv(original)(x), input_conv(control)(x)
    assert torch.equal(a[..., 1:-1], b[..., 1:-1])
    assert not torch.equal(a[..., 0], b[..., 0])
    assert not torch.equal(a[..., -1], b[..., -1])


@pytest.mark.parametrize('mode,native,changed', [
    ('E0', 'circular', 'replicate'), ('E1', 'replicate', 'circular'),
])
def test_native_padding_keeps_legacy_setting_but_control_has_an_independent_name(mode, native, changed):
    args = padding_args(mode)
    legacy = copy.deepcopy(args)
    del legacy.input_padding  # configurations saved before this option existed
    expected = build_setting(legacy, 0)
    assert build_setting(args, 0) == expected
    assert build_setting(padding_args(mode, native), 0) == expected
    new_name = build_setting(padding_args(mode, changed), 0)
    assert new_name != expected
    assert '_' + mode + 'P_cfg' in new_name
    a, b = vars(args).copy(), vars(padding_args(mode, changed)).copy()
    a.pop('input_padding'); b.pop('input_padding')
    assert a == b


@pytest.mark.parametrize('mode,padding', [('E0', 'replicate'), ('E1', 'circular')])
def test_padding_control_forward_backward_and_updated_layer_chain(mode, padding):
    torch.manual_seed(2025)
    model = Model(padding_args(mode, padding)).train()
    observed = []
    hooks = [block.register_forward_hook(lambda m, a, out: observed.append((a[0], out[0])))
             for block in model.blocks]
    x = torch.randn(2, 96, 10)
    y, aux = model(x)
    ((y - x).square().flatten(1).sum(1).mean() + aux).backward()
    assert y.shape == x.shape and torch.isfinite(y).all() and torch.isfinite(aux)
    assert input_conv(model).weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    expected = (2, 96, 64) if mode == 'E0' else (2, 96, 10, 16)
    assert len(observed) == 3
    assert all(a.shape == b.shape == expected for a, b in observed)
    assert all(observed[i][0] is observed[i-1][1] for i in range(1, 3))
    for hook in hooks:
        hook.remove()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x)[0], model(x)[0])


@pytest.mark.parametrize('mode', ['E0', 'E1'])
def test_default_and_explicit_native_outputs_are_bit_identical(mode):
    native = 'circular' if mode == 'E0' else 'replicate'
    torch.manual_seed(2025)
    implicit = Model(padding_args(mode)).eval()
    torch.manual_seed(2025)
    explicit = Model(padding_args(mode, native)).eval()
    x = torch.randn(2, 96, 10)
    with torch.no_grad():
        assert all(torch.equal(a, b) for a, b in zip(implicit(x), explicit(x)))
