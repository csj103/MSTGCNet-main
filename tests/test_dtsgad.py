import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models.DTSGAD import Model
from run import build_parser, normalize_args


def build_args(extra=None):
    argv = [
        "--model",
        "DTSGAD",
        "--use_gpu",
        "false",
        "--paper_strict",
        "false",
        "--seq_len",
        "32",
        "--winsize",
        "32",
        "--batch_size",
        "4",
        "--d_model",
        "16",
        "--d_ff",
        "32",
        "--e_layers",
        "2",
        "--n_heads",
        "2",
        "--attn_heads",
        "2",
        "--num_experts_list",
        "3",
        "3",
        "--patch_size_list",
        "4",
        "8",
        "16",
        "4",
        "8",
        "16",
        "--top_k",
        "2",
    ]
    if extra:
        argv.extend(extra)
    return normalize_args(build_parser().parse_args(argv))


def test_dtsgad_returns_probabilistic_dual_score_shapes():
    torch.manual_seed(2025)
    args = build_args()
    model = Model(args).train()
    inputs = torch.randn(4, 32, 10)

    outputs = model(inputs, return_dict=True)

    assert outputs["reconstruction"].shape == inputs.shape
    assert outputs["reconstruction_logvar"].shape == inputs.shape
    assert outputs["observation_score"].shape == (4, 32, 10)
    assert outputs["dynamic_score"].shape == (4, 32)
    assert outputs["total_score"].shape == (4, 32)
    assert outputs["balance_loss"].requires_grad
    assert torch.isfinite(outputs["total_score"]).all()


def test_dtsgad_training_loss_backpropagates_through_probabilistic_heads():
    torch.manual_seed(2025)
    args = build_args()
    model = Model(args).train()
    exp = object.__new__(Exp_Anomaly_Detection)
    exp.args = args
    inputs = torch.randn(4, 32, 10)

    outputs = model(inputs, return_dict=True)
    loss, parts = Exp_Anomaly_Detection._model_loss_from_output(
        outputs, inputs, args
    )
    loss.backward()

    assert parts["nll_loss"] > 0
    assert parts["dynamic_loss"] >= 0
    assert model.reconstruction_mu.weight.grad is not None
    assert model.reconstruction_logvar.weight.grad is not None
    assert model.posterior_mu.weight.grad is not None


def test_dtsgad_ablation_can_remove_spectral_and_dynamic_score():
    torch.manual_seed(2025)
    args = build_args(
        [
            "--disable_spectral",
            "true",
            "--disable_dynamic_score",
            "true",
            "--score_fusion",
            "obs",
        ]
    )
    model = Model(args).eval()
    inputs = torch.randn(2, 32, 10)

    with torch.no_grad():
        outputs = model(inputs, return_dict=True)

    obs_topk = torch.topk(outputs["observation_score"][:, :, :], k=3, dim=-1).values
    expected = obs_topk.mean(dim=-1)
    assert torch.allclose(outputs["dynamic_score"], torch.zeros_like(expected))
    assert torch.allclose(outputs["total_score"], expected, atol=1e-6)


def test_dtsgad_legacy_tuple_interface_matches_reconstruction_and_aux_loss():
    torch.manual_seed(2025)
    args = build_args()
    model = Model(args).train()
    inputs = torch.randn(2, 32, 10)

    reconstruction, aux_loss = model(inputs)
    outputs = model(inputs, return_dict=True)

    assert reconstruction.shape == inputs.shape
    assert aux_loss.ndim == 0
    assert torch.isfinite(aux_loss)
    assert outputs["balance_loss"].ndim == 0
