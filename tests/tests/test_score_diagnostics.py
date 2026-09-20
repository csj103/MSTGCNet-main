from pathlib import Path

import numpy as np
import pandas as pd

from utils.score_diagnostics import run_score_diagnostics


def test_score_diagnostics_writes_s1_s3_tables_and_s2_plots(tmp_path):
    result_dir = tmp_path / "result"
    component_dir = result_dir / "score_components"
    component_dir.mkdir(parents=True)

    labels = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    indices = np.arange(labels.size)
    np.save(result_dir / "test_labels.npy", labels)
    np.save(result_dir / "test_indices.npy", indices)
    np.save(result_dir / "threshold.npy", np.full(labels.size, 0.5))
    np.save(result_dir / "train_feature_error_scale.npy", np.array([0.1, 0.4]))
    np.save(component_dir / "s_obs_topk.npy", np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7]))
    np.save(component_dir / "s_dyn.npy", np.array([0.6, 0.4, 0.5, 0.3, 0.2, 0.1]))
    np.save(component_dir / "s_total.npy", np.array([0.2, 0.3, 1.0, 0.9, 0.4, 0.8]))
    np.save(
        component_dir / "s_obs_feature.npy",
        np.array(
            [
                [0.1, 0.3],
                [0.2, 0.2],
                [0.8, 0.9],
                [0.7, 0.8],
                [0.2, 0.4],
                [0.6, 0.7],
            ]
        ),
    )
    np.save(component_dir / "s_mse_topk.npy", np.array([0.1, 0.1, 0.9, 0.8, 0.2, 0.7]))
    np.save(component_dir / "s_stdres_topk.npy", np.array([0.2, 0.2, 0.8, 0.9, 0.3, 0.6]))
    np.save(component_dir / "s_unc_mean.npy", np.array([-1.0, -1.0, 3.0, 4.0, -6.0, 4.0]))
    np.save(component_dir / "s_nll_topk.npy", np.array([0.2, 0.3, 1.2, 1.0, 0.1, 0.9]))
    np.save(
        component_dir / "s_mse_feature.npy",
        np.array(
            [
                [0.1, 0.2],
                [0.1, 0.2],
                [0.9, 0.8],
                [0.8, 0.7],
                [0.2, 0.1],
                [0.7, 0.6],
            ]
        ),
    )
    np.save(
        component_dir / "s_stdres_feature.npy",
        np.array(
            [
                [0.2, 0.1],
                [0.2, 0.1],
                [0.8, 0.9],
                [0.9, 0.8],
                [0.3, 0.2],
                [0.6, 0.7],
            ]
        ),
    )
    np.save(
        component_dir / "s_unc_feature.npy",
        np.array(
            [
                [-1.0, -1.0],
                [-1.0, -1.0],
                [3.0, 4.0],
                [4.0, 4.0],
                [-6.0, -6.0],
                [4.0, 4.0],
            ]
        ),
    )
    np.save(
        component_dir / "s_nll_feature.npy",
        np.array(
            [
                [0.2, 0.2],
                [0.3, 0.3],
                [1.2, 1.1],
                [1.0, 0.9],
                [0.1, 0.1],
                [0.9, 0.8],
            ]
        ),
    )
    np.save(
        component_dir / "router_gates.npy",
        np.array(
            [
                [[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]],
                [[0.6, 0.3, 0.1], [0.3, 0.4, 0.3]],
                [[0.2, 0.6, 0.2], [0.2, 0.6, 0.2]],
                [[0.1, 0.7, 0.2], [0.1, 0.7, 0.2]],
                [[0.8, 0.1, 0.1], [0.4, 0.4, 0.2]],
                [[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]],
            ]
        ),
    )
    np.save(
        component_dir / "expert_pair_cosine.npy",
        np.array(
            [
                [[0.9, 0.8, 0.7], [0.95, 0.85, 0.75]],
                [[0.8, 0.7, 0.6], [0.94, 0.84, 0.74]],
                [[0.7, 0.6, 0.5], [0.93, 0.83, 0.73]],
                [[0.6, 0.5, 0.4], [0.92, 0.82, 0.72]],
                [[0.5, 0.4, 0.3], [0.91, 0.81, 0.71]],
                [[0.4, 0.3, 0.2], [0.90, 0.80, 0.70]],
            ]
        ),
    )
    np.save(
        component_dir / "expert_pre_graph_pair_cosine.npy",
        np.array(
            [
                [[0.4, 0.3, 0.2], [0.5, 0.4, 0.3]],
                [[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]],
                [[0.6, 0.5, 0.4], [0.7, 0.6, 0.5]],
                [[0.7, 0.6, 0.5], [0.8, 0.7, 0.6]],
                [[0.8, 0.7, 0.6], [0.9, 0.8, 0.7]],
                [[0.9, 0.8, 0.7], [0.95, 0.85, 0.75]],
            ]
        ),
    )
    np.save(
        component_dir / "expert_post_graph_pair_cosine.npy",
        np.array(
            [
                [[0.9, 0.8, 0.7], [0.95, 0.85, 0.75]],
                [[0.8, 0.7, 0.6], [0.94, 0.84, 0.74]],
                [[0.7, 0.6, 0.5], [0.93, 0.83, 0.73]],
                [[0.6, 0.5, 0.4], [0.92, 0.82, 0.72]],
                [[0.5, 0.4, 0.3], [0.91, 0.81, 0.71]],
                [[0.4, 0.3, 0.2], [0.90, 0.80, 0.70]],
            ]
        ),
    )
    graph_trace = np.zeros((6, 2, 5, 3), dtype=float)
    graph_trace[:, :, 0, :] = 0.5
    graph_trace[:, :, 1, :] = 0.6
    graph_trace[:, :, 2, :] = 1.0
    graph_trace[:, :, 3, :] = 0.9
    graph_trace[:, :, 4, :] = 0.95
    np.save(component_dir / "graph_trace_pair_cosine.npy", graph_trace)
    adjacency = np.zeros((6, 2, 3, 2, 2))
    adjacency[..., 0, 0] = 1.0
    adjacency[..., 1, 1] = 1.0
    adjacency[:, :, :, 0, 1] = 0.5
    np.save(component_dir / "adjacency.npy", adjacency)
    np.save(
        component_dir / "single_expert_mse_topk.npy",
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.1, 0.3],
                [0.9, 0.8, 0.7],
                [0.8, 0.7, 0.6],
                [0.1, 0.2, 0.1],
                [0.7, 0.8, 0.9],
            ]
        ),
    )
    np.save(
        component_dir / "single_expert_nll_topk.npy",
        np.array(
            [
                [0.2, 0.1, 0.3],
                [0.2, 0.2, 0.4],
                [1.0, 0.8, 0.7],
                [0.9, 0.7, 0.6],
                [0.1, 0.2, 0.1],
                [0.8, 0.9, 1.0],
            ]
        ),
    )
    (component_dir / "router_metadata.json").write_text(
        (
            '{"patch_size_list": [[8, 16, 32], [4, 8, 16]], '
            '"top_k": 2, "knn_k": 1, '
            '"graph_trace_positions": ["input", "after_adjacency", '
            '"residual_base", "after_residual", "after_norm"]}'
        ),
        encoding="utf-8",
    )

    meta = pd.DataFrame(
        {
            "source_file": [
                "normal.csv",
                "normal.csv",
                "carbonZ_2018-09-11-14-22-07_2_engine_failure.csv",
                "carbonZ_2018-09-11-14-22-07_2_engine_failure.csv",
                "carbonZ_2018-09-11-15-06-34_1_rudder_right_failure.csv",
                "carbonZ_2018-09-11-15-06-34_1_rudder_right_failure.csv",
            ],
            "fault_type": ["normal", "normal", "engine", "engine", "rudder", "rudder"],
            "fine_anomaly_type": [
                "normal",
                "normal",
                "Engine full power loss",
                "Engine full power loss",
                "Rudder stuck to right",
                "Rudder stuck to right",
            ],
            "segment_id": [0, 0, 1, 1, 2, 2],
            "time_sec": np.arange(labels.size, dtype=float),
            "label": labels,
        }
    )
    meta_path = tmp_path / "test_meta.csv"
    meta.to_csv(meta_path, index=False)

    output_dir = run_score_diagnostics(
        result_dir=result_dir,
        test_meta_path=meta_path,
        feature_names=["a", "b"],
        fault_types=[
            "Engine full power loss",
            "Rudder stuck to right",
        ],
    )

    assert (output_dir / "s1_component_auc.csv").exists()
    assert (output_dir / "s1_component_distribution_stats.csv").exists()
    assert (output_dir / "s1_obs_feature_auc.csv").exists()
    assert (output_dir / "m2_prediction_feature_auc.csv").exists()
    assert (output_dir / "m2_train_standardized_topk_auc.csv").exists()
    assert (output_dir / "s1_score_auc_by_anomaly_type.csv").exists()
    assert (output_dir / "s1_score_auc_by_anomaly_type.png").exists()
    assert (output_dir / "s4_probabilistic_score_auc.csv").exists()
    assert (output_dir / "s4_probabilistic_feature_auc.csv").exists()
    assert (output_dir / "s5_logvar_stats.csv").exists()
    assert (output_dir / "s6_mse_vs_nll_Engine_full_power_loss.png").exists()
    assert (output_dir / "r1_router_gate_stats.csv").exists()
    assert (output_dir / "r1_router_entropy_stats.csv").exists()
    assert (output_dir / "r2_expert_cosine_stats.csv").exists()
    assert (output_dir / "r3_single_expert_auc.csv").exists()
    assert (output_dir / "r5_expert_score_fusion_auc.csv").exists()
    assert (output_dir / "r6_single_expert_auc_by_anomaly_type.csv").exists()
    assert (output_dir / "r4_adjacency_stats.csv").exists()
    assert (output_dir / "r4_adjacency_block1_expert1_patch8.png").exists()
    assert (output_dir / "g1_pre_post_graph_cosine.csv").exists()
    assert (output_dir / "g4_graph_trace_cosine.csv").exists()
    assert (output_dir / "g4_graph_trace_summary.csv").exists()
    assert (output_dir / "s3_score_distribution.png").exists()
    assert (output_dir / "s2_timeseries_Engine_full_power_loss.png").exists()
    assert (output_dir / "s2_timeseries_Rudder_stuck_to_right.png").exists()
    assert (output_dir / "a6_event_phase_score_stats.csv").exists()

    auc_rows = pd.read_csv(output_dir / "s1_component_auc.csv")
    assert set(auc_rows["score_name"]) == {"S_obs_topk", "S_dyn", "S_total"}
    assert auc_rows.loc[auc_rows["score_name"] == "S_obs_topk", "ROC-AUC"].iloc[0] == 1.0

    prediction_feature_rows = pd.read_csv(output_dir / "m2_prediction_feature_auc.csv")
    assert set(prediction_feature_rows["feature"]) == {"a", "b"}
    assert prediction_feature_rows.loc[
        prediction_feature_rows["feature"] == "a",
        "ROC-AUC",
    ].iloc[0] == 1.0

    standardized_rows = pd.read_csv(output_dir / "m2_train_standardized_topk_auc.csv")
    assert set(standardized_rows["score_name"]) == {"M2_train_standardized_topk"}
    assert standardized_rows["ROC-AUC"].iloc[0] == 1.0
    assert standardized_rows["feature_scale_source"].iloc[0].endswith(
        "train_feature_error_scale.npy"
    )

    stats_rows = pd.read_csv(output_dir / "s1_component_distribution_stats.csv")
    assert {"mean", "median", "std", "P95", "P99"}.issubset(stats_rows.columns)

    type_auc_rows = pd.read_csv(output_dir / "s1_score_auc_by_anomaly_type.csv")
    assert {"anomaly_type", "score_name", "ROC-AUC", "PR-AUC"}.issubset(
        type_auc_rows.columns
    )
    assert set(type_auc_rows["anomaly_type"]) == {
        "Engine full power loss",
        "Rudder stuck to right",
    }

    prob_rows = pd.read_csv(output_dir / "s4_probabilistic_score_auc.csv")
    assert set(prob_rows["score_name"]) == {
        "S_MSE_topk",
        "S_stdres_topk",
        "S_unc_mean",
        "S_NLL_topk",
    }

    logvar_rows = pd.read_csv(output_dir / "s5_logvar_stats.csv")
    assert {"min", "max", "P5", "P95", "lower_clamp_ratio", "upper_clamp_ratio"}.issubset(
        logvar_rows.columns
    )

    router_rows = pd.read_csv(output_dir / "r1_router_gate_stats.csv")
    assert {"block", "expert", "patch_size", "label_group", "mean", "std"}.issubset(
        router_rows.columns
    )

    expert_rows = pd.read_csv(output_dir / "r3_single_expert_auc.csv")
    assert {"expert", "patch_path", "score_name", "ROC-AUC", "PR-AUC"}.issubset(
        expert_rows.columns
    )
    fusion_rows = pd.read_csv(output_dir / "r5_expert_score_fusion_auc.csv")
    assert {
        "score_name",
        "fusion",
        "ROC-AUC",
        "PR-AUC",
    }.issubset(fusion_rows.columns)
    assert {
        "E1",
        "E2",
        "E3",
        "Mean",
        "Max",
        "CurrentRouter",
    }.issubset(set(fusion_rows["fusion"]))

    type_expert_rows = pd.read_csv(output_dir / "r6_single_expert_auc_by_anomaly_type.csv")
    assert {
        "anomaly_type",
        "score_name",
        "expert",
        "patch_path",
        "ROC-AUC",
        "PR-AUC",
        "points",
        "anomaly_points",
    }.issubset(type_expert_rows.columns)
    assert set(type_expert_rows["anomaly_type"]) == {
        "Engine full power loss",
        "Rudder stuck to right",
    }

    g1_rows = pd.read_csv(output_dir / "g1_pre_post_graph_cosine.csv")
    assert set(g1_rows["position"]) == {"pre_graph", "post_graph"}
    assert {"block", "position", "expert_pair", "mean", "std"}.issubset(
        g1_rows.columns
    )

    g4_rows = pd.read_csv(output_dir / "g4_graph_trace_cosine.csv")
    assert set(g4_rows["position"]) == {
        "input",
        "after_adjacency",
        "residual_base",
        "after_residual",
        "after_norm",
    }
    assert {"block", "position", "expert_pair", "mean", "std"}.issubset(
        g4_rows.columns
    )
    g4_summary = pd.read_csv(output_dir / "g4_graph_trace_summary.csv")
    assert {"Position", "B1 cosine", "B2 cosine"}.issubset(g4_summary.columns)

    phase_rows = pd.read_csv(output_dir / "a6_event_phase_score_stats.csv")
    assert {
        "anomaly_type",
        "score_name",
        "phase",
        "mean",
        "median",
        "P95",
    }.issubset(phase_rows.columns)
    assert {"pre_normal", "early", "middle", "late"}.issubset(
        set(phase_rows["phase"])
    )


def test_score_diagnostics_can_plot_overlap_mean_energy_without_components(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir(parents=True)
    labels = np.array([0, 0, 1, 1, 0, 0], dtype=int)
    indices = np.arange(labels.size)
    np.save(result_dir / "test_labels.npy", labels)
    np.save(result_dir / "test_indices.npy", indices)
    np.save(result_dir / "test_energy.npy", np.array([0.1, 0.2, 0.9, 0.8, 0.2, 0.1]))
    np.save(result_dir / "threshold.npy", np.full(labels.size, 0.5))

    meta = pd.DataFrame(
        {
            "fine_anomaly_type": [
                "normal",
                "normal",
                "Engine full power loss",
                "Engine full power loss",
                "normal",
                "normal",
            ],
            "segment_id": [0, 0, 1, 1, 1, 1],
            "label": labels,
        }
    )
    meta_path = tmp_path / "test_meta.csv"
    meta.to_csv(meta_path, index=False)

    output_dir = run_score_diagnostics(
        result_dir=result_dir,
        test_meta_path=meta_path,
        fault_types=["Engine full power loss"],
    )

    auc_rows = pd.read_csv(output_dir / "s1_component_auc.csv")
    assert set(auc_rows["score_name"]) == {"S_total"}
    assert (output_dir / "s2_timeseries_Engine_full_power_loss.png").exists()
    assert (output_dir / "s3_score_distribution.png").exists()
