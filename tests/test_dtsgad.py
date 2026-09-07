import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from models.DTSGAD import Model, TemporalVariableGraphExpert
from run import build_parser, build_setting, normalize_args


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


def test_dtsgad_forward_masks_input_but_scores_original_target():
    torch.manual_seed(2025)
    args = build_args(["--revin", "0"])
    model = Model(args).eval()
    inputs = torch.randn(2, 32, 10)
    input_mask = torch.ones_like(inputs)
    input_mask[:, :, 0] = 0.0

    with torch.no_grad():
        outputs = model(inputs, input_mask=input_mask, return_dict=True)

    assert torch.allclose(outputs["target_norm"], inputs)
    assert torch.allclose(outputs["input_norm"][:, :, 0], torch.zeros_like(inputs[:, :, 0]))
    assert torch.allclose(outputs["input_norm"][:, :, 1:], inputs[:, :, 1:])


def test_dtsgad_loss_can_focus_only_on_masked_positions():
    args = build_args(
        [
            "--dynamic_loss_weight",
            "0.0",
            "--last_loss_weight",
            "0.0",
            "--balance_loss_weight",
            "0.0",
        ]
    )
    observation_score = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ]
    )
    loss_mask = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )

    loss, parts = Exp_Anomaly_Detection._model_loss_from_output(
        {"observation_score": observation_score},
        torch.zeros_like(observation_score),
        args,
        loss_mask=loss_mask,
    )

    assert torch.allclose(loss, torch.tensor((10.0 + 20.0 + 3.0 + 4.0) / 4.0))
    assert parts["reconstruction_loss"] == (10.0 + 20.0 + 3.0 + 4.0) / 4.0


def test_random_channel_mask_hides_whole_variables_per_window():
    torch.manual_seed(2025)
    batch = torch.ones(5, 7, 10)

    input_mask, loss_mask = Exp_Anomaly_Detection._random_channel_masks(
        batch,
        mask_ratio=0.2,
    )

    assert input_mask.shape == batch.shape
    assert loss_mask.shape == batch.shape
    assert torch.allclose(input_mask + loss_mask, torch.ones_like(batch))
    assert torch.equal(loss_mask[:, 0, :], loss_mask[:, -1, :])
    assert torch.all(loss_mask[:, 0, :].sum(dim=-1) == 2)


def test_channelwise_masked_scoring_masks_each_feature_once():
    args = build_args(
        [
            "--mask_ratio",
            "0.2",
            "--mask_eval_mode",
            "channelwise",
            "--score_fusion",
            "obs",
            "--obs_topk",
            "1",
            "--dynamic_score_weight",
            "0.05",
        ]
    )
    exp = object.__new__(Exp_Anomaly_Detection)
    exp.args = args
    calls = []
    batch = torch.ones(2, 4, 3)
    marks = torch.zeros_like(batch)

    def fake_forward(batch_x, batch_m, **kwargs):
        input_mask = kwargs["input_mask"]
        calls.append(input_mask.detach().clone())
        masked_feature = int(torch.argmin(input_mask[0, 0]).item())
        observation = torch.zeros_like(batch_x)
        observation[:, :, masked_feature] = float(masked_feature + 1)
        dynamic = torch.full(batch_x.shape[:2], float(masked_feature + 1))
        return {
            "reconstruction": batch_x * 0.0,
            "reconstruction_norm": batch_x * 0.0,
            "reconstruction_logvar": batch_x * 0.0,
            "target_norm": batch_x,
            "observation_score": observation,
            "dynamic_score": dynamic,
            "total_score": observation.mean(dim=-1),
            "balance_loss": torch.zeros((), dtype=batch_x.dtype),
        }

    exp._forward_model = fake_forward

    outputs = exp._score_forward_model(batch, marks)

    assert len(calls) == 3
    for feature_index, input_mask in enumerate(calls):
        assert torch.all(input_mask[:, :, feature_index] == 0.0)
        assert torch.all(input_mask[:, :, :feature_index] == 1.0)
        assert torch.all(input_mask[:, :, feature_index + 1 :] == 1.0)
    assert outputs["observation_score"][0, 0].tolist() == [1.0, 2.0, 3.0]
    assert outputs["total_score"][0, 0].item() == 3.0


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


def test_dtsgad_next_step_prediction_outputs_single_target_score():
    torch.manual_seed(2025)
    args = build_args(
        [
            "--dtsgad_objective",
            "next_step_prediction",
            "--revin",
            "0",
        ]
    )
    model = Model(args).eval()
    inputs = torch.randn(4, 32, 10)
    target_next = torch.randn(4, 10)

    with torch.no_grad():
        outputs = model(inputs, target_next=target_next, return_dict=True)

    assert outputs["prediction"].shape == target_next.shape
    assert outputs["prediction_score"].shape == target_next.shape
    assert outputs["observation_score"].shape == (4, 1, 10)
    assert outputs["total_score"].shape == (4, 1)
    assert torch.allclose(outputs["target_next_norm"], target_next)


def test_dtsgad_next_step_prediction_loss_is_plain_mse():
    args = build_args(["--dtsgad_objective", "next_step_prediction"])
    prediction_score = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
    loss, parts = Exp_Anomaly_Detection._model_loss_from_output(
        {
            "prediction_score": prediction_score,
            "dynamic_score": torch.full((2, 1), 100.0),
            "balance_loss": torch.tensor(100.0),
        },
        torch.zeros(2, 2),
        args,
    )

    assert torch.allclose(loss, prediction_score.mean())
    assert parts["reconstruction_loss"] == prediction_score.mean().item()
    assert parts["dynamic_loss"] == 0.0
    assert parts["balance_loss"] == 100.0


def test_next_step_prediction_normalization_disables_auxiliary_losses():
    args = build_args(
        [
            "--dtsgad_objective",
            "next_step_prediction",
            "--dynamic_loss_weight",
            "0.2",
            "--last_loss_weight",
            "0.2",
            "--balance_loss_weight",
            "0.2",
        ]
    )

    assert args.disable_probabilistic is True
    assert args.disable_dynamic_score is True
    assert args.score_fusion == "obs"
    assert args.dynamic_loss_weight == 0.0
    assert args.last_loss_weight == 0.0
    assert args.balance_loss_weight == 0.0


def test_next_step_causal_last_scores_align_to_future_target_index():
    args = build_args(
        [
            "--dtsgad_objective",
            "next_step_prediction",
            "--seq_len",
            "3",
            "--winsize",
            "3",
            "--patch_size_list",
            "1",
            "2",
            "3",
            "1",
            "2",
            "3",
        ]
    )
    exp = object.__new__(Exp_Anomaly_Detection)
    exp.args = args
    exp.device = torch.device("cpu")

    class DummyModel:
        def eval(self):
            return self

    exp.model = DummyModel()
    seen_targets = []

    def fake_score_forward(batch_x, batch_m, **kwargs):
        target_next = kwargs["target_next"]
        seen_targets.extend(target_next[:, 0].tolist())
        prediction_score = target_next.pow(2)
        return {
            "prediction": target_next * 0.0,
            "prediction_norm": target_next * 0.0,
            "target_next_norm": target_next,
            "target_norm": target_next,
            "prediction_score": prediction_score,
            "observation_score": prediction_score.unsqueeze(1),
            "dynamic_score": torch.zeros(target_next.size(0), 1),
            "total_score": torch.topk(
                prediction_score,
                k=1,
                dim=-1,
            ).values.mean(dim=-1, keepdim=True),
            "balance_loss": torch.zeros(()),
        }

    exp._score_forward_model = fake_score_forward
    values = torch.arange(6, dtype=torch.float32).view(6, 1).numpy()
    marks = values.copy()
    labels = torch.tensor([0, 0, 0, 1, 0, 1], dtype=torch.float32).numpy()

    scores, targets, indices, _ = exp._causal_last_scores(
        values,
        marks,
        labels,
        windows=[0, 1],
    )

    assert indices.tolist() == [3, 4]
    assert targets.tolist() == [1.0, 0.0]
    assert seen_targets == [3.0, 4.0]
    assert scores.tolist() == [9.0, 16.0]


def test_dtsgad_causal_last_component_extraction_uses_last_timestamp():
    observation = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    dynamic = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    total = dynamic + 100
    output = {
        "observation_score": observation,
        "dynamic_score": dynamic,
        "total_score": total,
    }

    components = Exp_Anomaly_Detection._dtsgad_causal_last_components(
        output,
        obs_topk=2,
    )

    assert components["s_obs_feature"].shape == (2, 3)
    assert components["s_obs_feature"][0].tolist() == [9.0, 10.0, 11.0]
    assert components["s_obs_topk"].tolist() == [10.5, 22.5]
    assert components["s_dyn"].tolist() == [3.0, 7.0]
    assert components["s_total"].tolist() == [103.0, 107.0]


def test_dtsgad_causal_last_component_extraction_decomposes_probabilistic_score():
    target = torch.tensor([[[1.0, 2.0], [3.0, 5.0]]])
    mu = torch.tensor([[[0.0, 2.0], [1.0, 4.0]]])
    logvar = torch.tensor([[[0.0, 1.0], [-6.0, 4.0]]])
    variance = logvar.exp()
    mse = (target - mu).pow(2)
    stdres = mse / variance
    nll = 0.5 * (stdres + logvar + torch.log(torch.tensor(2.0 * torch.pi)))
    output = {
        "target_norm": target,
        "reconstruction_norm": mu,
        "reconstruction_logvar": logvar,
        "observation_score": nll,
    }

    components = Exp_Anomaly_Detection._dtsgad_causal_last_components(
        output,
        obs_topk=1,
    )

    assert components["s_mse_feature"].tolist() == [[4.0, 1.0]]
    assert components["s_stdres_feature"].shape == (1, 2)
    assert components["s_unc_feature"].tolist() == [[-6.0, 4.0]]
    assert components["s_nll_feature"].shape == (1, 2)
    assert components["s_mse_topk"].tolist() == [4.0]
    assert components["s_stdres_topk"].tolist()[0] > 1000.0
    assert components["s_unc_mean"].tolist() == [-1.0]
    assert components["s_nll_topk"].tolist()[0] > 700.0


def test_dtsgad_forward_can_collect_router_expert_graph_diagnostics():
    torch.manual_seed(2025)
    args = build_args()
    model = Model(args).eval()
    inputs = torch.randn(2, 32, 10)

    with torch.no_grad():
        outputs = model(inputs, return_dict=True, collect_diagnostics=True)

    diagnostics = outputs["router_diagnostics"]
    assert len(diagnostics) == args.e_layers
    first = diagnostics[0]
    assert first["gates"].shape == (2, 3)
    assert first["expert_pair_cosine"].shape == (2, 3)
    assert first["expert_pre_graph_pair_cosine"].shape == (2, 3)
    assert first["expert_post_graph_pair_cosine"].shape == (2, 3)
    assert first["graph_trace_pair_cosine"].shape == (2, 5, 3)
    assert first["graph_trace_positions"] == [
        "input",
        "after_adjacency",
        "residual_base",
        "after_residual",
        "after_norm",
    ]
    assert first["adjacency"].shape == (3, 10, 10)
    assert torch.allclose(first["gates"].sum(dim=-1), torch.ones(2), atol=1e-6)


def test_dtsgad_graph_off_keeps_pre_and_post_graph_similarity_equal():
    torch.manual_seed(2025)
    args = build_args(["--graph_mode", "off"])
    model = Model(args).eval()
    inputs = torch.randn(2, 32, 10)

    with torch.no_grad():
        outputs = model(inputs, return_dict=True, collect_diagnostics=True)

    first = outputs["router_diagnostics"][0]
    assert torch.allclose(
        first["expert_pre_graph_pair_cosine"],
        first["expert_post_graph_pair_cosine"],
        atol=1e-6,
    )


def test_dtsgad_identity_graph_uses_identity_adjacency_but_keeps_graph_block():
    torch.manual_seed(2025)
    args = build_args(["--graph_mode", "identity"])
    model = Model(args).eval()
    inputs = torch.randn(2, 32, 10)

    with torch.no_grad():
        outputs = model(inputs, return_dict=True, collect_diagnostics=True)

    first = outputs["router_diagnostics"][0]
    expected = torch.eye(args.enc_in).expand(3, -1, -1)
    assert torch.allclose(first["adjacency"], expected, atol=1e-6)
    assert not torch.allclose(
        first["expert_pre_graph_pair_cosine"],
        first["expert_post_graph_pair_cosine"],
        atol=1e-4,
    )


def test_dtsgad_expert_residual_uses_pre_graph_as_residual_source():
    torch.manual_seed(2025)
    expert = TemporalVariableGraphExpert(
        seq_len=8,
        num_vars=3,
        d_model=4,
        d_ff=8,
        patch_size=4,
        attn_heads=2,
        knn_k=1,
        dropout=0.0,
        graph_mode="identity",
        graph_residual_mode="expert",
    ).eval()
    inputs = torch.randn(2, 8, 3, 4)

    with torch.no_grad():
        _, _, _, _, trace = expert(inputs, collect_graph_states=True)

    assert torch.allclose(
        trace["after_residual"],
        trace["input"] + trace["after_adjacency"],
        atol=1e-6,
    )
    assert not torch.allclose(
        trace["after_residual"],
        trace["residual_base"] + trace["after_adjacency"],
        atol=1e-4,
    )


def test_dtsgad_mixed_residual_interpolates_shared_and_expert_sources():
    torch.manual_seed(2025)
    alpha = 0.25
    expert = TemporalVariableGraphExpert(
        seq_len=8,
        num_vars=3,
        d_model=4,
        d_ff=8,
        patch_size=4,
        attn_heads=2,
        knn_k=1,
        dropout=0.0,
        graph_mode="identity",
        graph_residual_mode="mixed",
        graph_residual_alpha=alpha,
    ).eval()
    inputs = torch.randn(2, 8, 3, 4)

    with torch.no_grad():
        _, _, _, _, trace = expert(inputs, collect_graph_states=True)

    expected = (
        alpha * trace["residual_base"]
        + (1.0 - alpha) * trace["input"]
        + trace["after_adjacency"]
    )
    assert torch.allclose(trace["after_residual"], expected, atol=1e-6)


def test_default_shared_residual_keeps_existing_current_graph_checkpoint_setting():
    args = normalize_args(
        build_parser().parse_args(
            [
                "--model",
                "DTSGAD",
                "--use_gpu",
                "false",
                "--num_workers",
                "0",
                "--paper_strict",
                "false",
                "--implementation_tag",
                "dtsgad_threshold_sweep_v1",
                "--router_diagnostics",
                "true",
                "--single_expert_diagnostics",
                "false",
                "--threshold_diagnostics",
                "false",
                "--graph_residual_mode",
                "shared",
            ]
        )
    )

    assert build_setting(args, 0).endswith("_cfg704eb8909c_0")


def test_latent_diagnostics_keeps_current_full_checkpoint_setting():
    args = normalize_args(
        build_parser().parse_args(
            [
                "--model",
                "DTSGAD",
                "--use_gpu",
                "false",
                "--num_workers",
                "0",
                "--paper_strict",
                "false",
                "--implementation_tag",
                "dtsgad_threshold_sweep_v1",
                "--latent_diagnostics",
                "true",
                "--router_diagnostics",
                "true",
                "--single_expert_diagnostics",
                "false",
                "--threshold_diagnostics",
                "false",
            ]
        )
    )

    assert build_setting(args, 0).endswith("_cfg704eb8909c_0")


def test_mixed_residual_alpha_changes_checkpoint_setting():
    base_argv = [
        "--model",
        "DTSGAD",
        "--use_gpu",
        "false",
        "--num_workers",
        "0",
        "--paper_strict",
        "false",
        "--implementation_tag",
        "dtsgad_alpha_residual",
        "--router_diagnostics",
        "true",
        "--single_expert_diagnostics",
        "false",
        "--threshold_diagnostics",
        "false",
        "--graph_residual_mode",
        "mixed",
    ]
    args_025 = normalize_args(
        build_parser().parse_args(base_argv + ["--graph_residual_alpha", "0.25"])
    )
    args_050 = normalize_args(
        build_parser().parse_args(base_argv + ["--graph_residual_alpha", "0.5"])
    )

    setting_025 = build_setting(args_025, 0)
    setting_050 = build_setting(args_050, 0)

    assert setting_025 != setting_050
    assert "_cfg" in setting_025
    assert "_cfg" in setting_050


def test_default_learned_graph_keeps_existing_current_graph_checkpoint_setting():
    args = normalize_args(
        build_parser().parse_args(
            [
                "--model",
                "DTSGAD",
                "--use_gpu",
                "false",
                "--num_workers",
                "0",
                "--paper_strict",
                "false",
                "--implementation_tag",
                "dtsgad_threshold_sweep_v1",
                "--router_diagnostics",
                "true",
                "--single_expert_diagnostics",
                "false",
                "--threshold_diagnostics",
                "false",
            ]
        )
    )

    assert build_setting(args, 0).endswith("_cfg704eb8909c_0")


def test_dtsgad_component_extraction_saves_router_diagnostics():
    observation = torch.ones(2, 4, 3)
    output = {
        "observation_score": observation,
        "router_diagnostics": [
            {
                "gates": torch.tensor([[0.7, 0.3], [0.2, 0.8]]),
                "expert_pair_cosine": torch.tensor([[0.9], [0.8]]),
                "expert_pre_graph_pair_cosine": torch.tensor([[0.4], [0.5]]),
                "expert_post_graph_pair_cosine": torch.tensor([[0.7], [0.6]]),
                "graph_trace_pair_cosine": torch.ones(2, 5, 1),
                "adjacency": torch.ones(2, 3, 3),
            },
            {
                "gates": torch.tensor([[0.6, 0.4], [0.1, 0.9]]),
                "expert_pair_cosine": torch.tensor([[0.7], [0.6]]),
                "expert_pre_graph_pair_cosine": torch.tensor([[0.3], [0.2]]),
                "expert_post_graph_pair_cosine": torch.tensor([[0.8], [0.9]]),
                "graph_trace_pair_cosine": torch.zeros(2, 5, 1),
                "adjacency": torch.zeros(2, 3, 3),
            },
        ],
    }

    components = Exp_Anomaly_Detection._dtsgad_causal_last_components(
        output,
        obs_topk=1,
    )

    assert components["router_gates"].shape == (2, 2, 2)
    assert components["expert_pair_cosine"].shape == (2, 2, 1)
    assert components["expert_pre_graph_pair_cosine"].shape == (2, 2, 1)
    assert components["expert_post_graph_pair_cosine"].shape == (2, 2, 1)
    assert components["graph_trace_pair_cosine"].shape == (2, 2, 5, 1)
    assert components["adjacency"].shape == (2, 2, 2, 3, 3)


def test_latent_center_and_scores_use_last_posterior_state():
    posterior_mu = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 0.0], [3.0, 1.0]],
        ]
    )

    latents = Exp_Anomaly_Detection._latent_vectors_from_output(
        {"posterior_mu": posterior_mu}
    )
    center = Exp_Anomaly_Detection._latent_center(latents)
    scores = Exp_Anomaly_Detection._latent_distance_scores(latents, center)

    assert latents.tolist() == [[1.0, 1.0], [3.0, 1.0]]
    assert center.tolist() == [2.0, 1.0]
    assert scores.tolist() == [1.0, 1.0]


def test_latent_normality_auc_table_reports_global_auc():
    labels = torch.tensor([0, 0, 1, 1]).numpy()
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9]).numpy()

    table = Exp_Anomaly_Detection._latent_metrics_table(labels, scores)

    assert table[0]["score_name"] == "S_latent_center_l2"
    assert table[0]["ROC-AUC"] == 1.0
    assert table[0]["PR-AUC"] == 1.0
