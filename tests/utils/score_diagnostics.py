import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from utils.alfa_anomaly_types import (
    anomaly_type_sort_key,
    canonical_anomaly_type_from_row,
)
from utils.score_postprocess import aggregate_feature_errors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_FAULT_TYPES = [
    "Engine full power loss",
    "Rudder stuck to right",
    "Both aileron stuck at zero",
]


def _safe_auc(labels, scores, metric):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    if metric == "roc":
        return float(roc_auc_score(labels, scores))
    return float(average_precision_score(labels, scores))


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "P95": float("nan"),
            "P99": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "P95": float(np.percentile(values, 95)),
        "P99": float(np.percentile(values, 99)),
    }


def _slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _load_array(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def _optional_array(path):
    path = Path(path)
    return np.load(path) if path.exists() else None


def load_saved_scores(result_dir):
    result_dir = Path(result_dir)
    component_dir = result_dir / "score_components"
    if (component_dir / "s_total.npy").exists():
        scores = {"S_total": _load_array(component_dir / "s_total.npy")}
        if (component_dir / "s_obs_topk.npy").exists():
            scores = {
                "S_obs_topk": _load_array(component_dir / "s_obs_topk.npy"),
                **scores,
            }
        if (component_dir / "s_dyn.npy").exists():
            scores["S_dyn"] = _load_array(component_dir / "s_dyn.npy")
    else:
        scores = {"S_total": _load_array(result_dir / "test_energy.npy")}
    obs_feature_path = component_dir / "s_obs_feature.npy"
    obs_feature = _load_array(obs_feature_path) if obs_feature_path.exists() else None
    labels = _load_array(result_dir / "test_labels.npy").astype(int)
    labels[labels != 0] = 1
    indices = _load_array(result_dir / "test_indices.npy").astype(int)
    threshold = _load_array(result_dir / "threshold.npy")
    if threshold.reshape(-1).size == 1:
        threshold = np.full(labels.shape[0], float(threshold.reshape(-1)[0]))
    return scores, obs_feature, labels, indices, threshold


def load_probabilistic_scores(result_dir):
    component_dir = Path(result_dir) / "score_components"
    aggregate = {
        "S_MSE_topk": _optional_array(component_dir / "s_mse_topk.npy"),
        "S_stdres_topk": _optional_array(component_dir / "s_stdres_topk.npy"),
        "S_unc_mean": _optional_array(component_dir / "s_unc_mean.npy"),
        "S_NLL_topk": _optional_array(component_dir / "s_nll_topk.npy"),
    }
    feature = {
        "S_MSE": _optional_array(component_dir / "s_mse_feature.npy"),
        "S_stdres": _optional_array(component_dir / "s_stdres_feature.npy"),
        "S_unc_logvar": _optional_array(component_dir / "s_unc_feature.npy"),
        "S_NLL": _optional_array(component_dir / "s_nll_feature.npy"),
    }
    if any(value is None for value in aggregate.values()):
        aggregate = {}
    if any(value is None for value in feature.values()):
        feature = {}
    return aggregate, feature


def load_prediction_feature_errors(result_dir):
    component_dir = Path(result_dir) / "score_components"
    return _optional_array(component_dir / "s_mse_feature.npy")


def load_train_feature_error_scale(result_dir):
    result_dir = Path(result_dir)
    for filename in ("train_feature_error_scale.npy", "feature_scale.npy"):
        path = result_dir / filename
        if path.exists():
            return np.load(path), str(path)
    return None, None


def load_router_metadata(result_dir):
    metadata_path = Path(result_dir) / "score_components" / "router_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_router_scores(result_dir):
    component_dir = Path(result_dir) / "score_components"
    router = {
        "router_gates": _optional_array(component_dir / "router_gates.npy"),
        "expert_pair_cosine": _optional_array(
            component_dir / "expert_pair_cosine.npy"
        ),
        "expert_pre_graph_pair_cosine": _optional_array(
            component_dir / "expert_pre_graph_pair_cosine.npy"
        ),
        "expert_post_graph_pair_cosine": _optional_array(
            component_dir / "expert_post_graph_pair_cosine.npy"
        ),
        "graph_trace_pair_cosine": _optional_array(
            component_dir / "graph_trace_pair_cosine.npy"
        ),
        "adjacency": _optional_array(component_dir / "adjacency.npy"),
        "single_expert_mse_topk": _optional_array(
            component_dir / "single_expert_mse_topk.npy"
        ),
        "single_expert_nll_topk": _optional_array(
            component_dir / "single_expert_nll_topk.npy"
        ),
    }
    return router


def load_feature_names(data_root, obs_feature=None, feature_names=None):
    if feature_names:
        return list(feature_names)
    data_root = Path(data_root) if data_root is not None else None
    metadata_path = data_root / "metadata.json" if data_root is not None else None
    if metadata_path is not None and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            names = json.load(file).get("features", [])
        if obs_feature is None or len(names) == obs_feature.shape[1]:
            return names
    if obs_feature is None:
        return []
    return ["feature_{}".format(index) for index in range(obs_feature.shape[1])]


def load_aligned_meta(test_meta_path, score_indices):
    if test_meta_path is None:
        return None
    meta = pd.read_csv(test_meta_path)
    score_indices = np.asarray(score_indices, dtype=int)
    if score_indices.size == 0:
        return meta.iloc[[]].copy()
    if score_indices.min() < 0 or score_indices.max() >= len(meta):
        raise IndexError("test_indices.npy cannot align with test_meta.csv")
    aligned = meta.iloc[score_indices].copy().reset_index(drop=True)
    if "fine_anomaly_type" not in aligned.columns:
        aligned["fine_anomaly_type"] = aligned.apply(
            canonical_anomaly_type_from_row,
            axis=1,
        )
    return aligned


def _patch_size(metadata, block_index, expert_index):
    patch_sizes = metadata.get("patch_size_list", [])
    if block_index < len(patch_sizes) and expert_index < len(patch_sizes[block_index]):
        return patch_sizes[block_index][expert_index]
    return expert_index + 1


def _expert_path(metadata, expert_index):
    patch_sizes = metadata.get("patch_size_list", [])
    path = []
    for block_sizes in patch_sizes:
        if expert_index < len(block_sizes):
            path.append(str(block_sizes[expert_index]))
    return "-".join(path) if path else "expert_{}".format(expert_index + 1)


def _expert_pairs_from_pair_count(pair_count):
    expert_count = 1
    while expert_count * (expert_count - 1) // 2 < pair_count:
        expert_count += 1
    pairs = []
    for left in range(expert_count):
        for right in range(left + 1, expert_count):
            pairs.append((left, right))
    return pairs[:pair_count]


def component_auc_table(scores, labels):
    rows = []
    for name, values in scores.items():
        rows.append(
            {
                "score_name": name,
                "ROC-AUC": _safe_auc(labels, values, "roc"),
                "PR-AUC": _safe_auc(labels, values, "pr"),
            }
        )
    return pd.DataFrame(rows)


def distribution_stats_table(scores, labels):
    labels = np.asarray(labels).astype(int)
    rows = []
    groups = [("normal", labels == 0), ("anomaly", labels == 1)]
    for name, values in scores.items():
        for group_name, mask in groups:
            rows.append(
                {
                    "score_name": name,
                    "label_group": group_name,
                    **_stats(np.asarray(values)[mask]),
                }
            )
    return pd.DataFrame(rows)


def score_auc_by_anomaly_type_table(scores, labels, aligned_meta):
    if aligned_meta is None or "fine_anomaly_type" not in aligned_meta.columns:
        return pd.DataFrame()
    labels = np.asarray(labels).astype(int)
    anomaly_types = aligned_meta["fine_anomaly_type"].astype(str).to_numpy()
    fault_types = [
        name
        for name in sorted(set(anomaly_types), key=anomaly_type_sort_key)
        if name != "normal" and np.any((anomaly_types == name) & (labels == 1))
    ]
    rows = []
    for fault_type in fault_types:
        mask = anomaly_types == fault_type
        masked_labels = labels[mask]
        for score_name, values in scores.items():
            rows.append(
                {
                    "anomaly_type": fault_type,
                    "score_name": score_name,
                    "ROC-AUC": _safe_auc(masked_labels, values[mask], "roc"),
                    "PR-AUC": _safe_auc(masked_labels, values[mask], "pr"),
                    "points": int(mask.sum()),
                    "anomaly_points": int(masked_labels.sum()),
                }
            )
    return pd.DataFrame(rows)


def event_phase_score_stats_table(scores, labels, aligned_meta):
    if aligned_meta is None or "fine_anomaly_type" not in aligned_meta.columns:
        return pd.DataFrame()
    labels = np.asarray(labels).astype(int)
    labels[labels != 0] = 1
    rows = []
    if "segment_id" in aligned_meta.columns:
        segment_values = aligned_meta["segment_id"].to_numpy()
        boundaries = np.flatnonzero(
            np.r_[True, segment_values[1:] != segment_values[:-1], True]
        )
        spans = list(zip(boundaries[:-1], boundaries[1:]))
    else:
        spans = [(0, len(labels))]

    for segment_start, segment_end in spans:
        local = labels[segment_start:segment_end]
        changes = np.diff(np.r_[0, local, 0])
        starts = np.flatnonzero(changes == 1) + segment_start
        ends = np.flatnonzero(changes == -1) + segment_start
        for event_number, (start, end) in enumerate(zip(starts, ends), start=1):
            fault_type = str(aligned_meta.iloc[start]["fine_anomaly_type"])
            if fault_type == "normal":
                continue
            event_len = max(1, end - start)
            pre_start = max(segment_start, start - event_len)
            pre_indices = np.arange(pre_start, start)
            pre_indices = pre_indices[labels[pre_indices] == 0]
            phase_indices = {
                "pre_normal": pre_indices,
                "early": np.array_split(np.arange(start, end), 3)[0],
                "middle": np.array_split(np.arange(start, end), 3)[1],
                "late": np.array_split(np.arange(start, end), 3)[2],
            }
            segment_id = (
                aligned_meta.iloc[start]["segment_id"]
                if "segment_id" in aligned_meta.columns
                else 0
            )
            for score_name, values in scores.items():
                values = np.asarray(values, dtype=np.float64)
                for phase, indices in phase_indices.items():
                    rows.append(
                        {
                            "segment_id": segment_id,
                            "event_number": event_number,
                            "anomaly_type": fault_type,
                            "score_name": score_name,
                            "phase": phase,
                            **_stats(values[indices]),
                        }
                    )
    return pd.DataFrame(rows)


def plot_score_auc_by_anomaly_type(type_auc_table, output_path):
    if type_auc_table.empty:
        return None
    table = type_auc_table[type_auc_table["score_name"] == "S_total"]
    if table.empty:
        table = type_auc_table
    fig, axis = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(table))
    axis.bar(x - 0.18, table["ROC-AUC"], width=0.36, label="ROC-AUC")
    axis.bar(x + 0.18, table["PR-AUC"], width=0.36, label="PR-AUC")
    axis.set_xticks(x)
    axis.set_xticklabels(table["anomaly_type"], rotation=30, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("AUC")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def router_gate_stats_table(router_gates, labels, metadata):
    labels = np.asarray(labels).astype(int)
    rows = []
    num_blocks = router_gates.shape[1]
    num_experts = router_gates.shape[2]
    for block_index in range(num_blocks):
        for expert_index in range(num_experts):
            values = router_gates[:, block_index, expert_index]
            for group_name, mask in [("normal", labels == 0), ("anomaly", labels == 1)]:
                rows.append(
                    {
                        "block": block_index + 1,
                        "expert": expert_index + 1,
                        "patch_size": _patch_size(metadata, block_index, expert_index),
                        "label_group": group_name,
                        **_stats(values[mask]),
                        "min": float(np.min(values[mask])) if mask.any() else float("nan"),
                        "max": float(np.max(values[mask])) if mask.any() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def router_entropy_stats_table(router_gates, labels):
    labels = np.asarray(labels).astype(int)
    entropy = -np.sum(
        router_gates * np.log(np.clip(router_gates, 1e-12, None)),
        axis=-1,
    )
    rows = []
    for block_index in range(entropy.shape[1]):
        for group_name, mask in [("normal", labels == 0), ("anomaly", labels == 1)]:
            rows.append(
                {
                    "block": block_index + 1,
                    "label_group": group_name,
                    **_stats(entropy[mask, block_index]),
                    "min": float(np.min(entropy[mask, block_index]))
                    if mask.any()
                    else float("nan"),
                    "max": float(np.max(entropy[mask, block_index]))
                    if mask.any()
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def router_summary_table(router_gates):
    rows = []
    max_entropy = float(np.log(router_gates.shape[-1]))
    entropy = -np.sum(
        router_gates * np.log(np.clip(router_gates, 1e-12, None)),
        axis=-1,
    )
    for block_index in range(router_gates.shape[1]):
        mean_gates = router_gates[:, block_index].mean(axis=0)
        mean_entropy = float(entropy[:, block_index].mean())
        normalized_entropy = mean_entropy / max_entropy if max_entropy > 0 else float("nan")
        rows.append(
            {
                "block": block_index + 1,
                "mean_entropy": mean_entropy,
                "max_entropy": max_entropy,
                "normalized_entropy": normalized_entropy,
                "max_mean_gate": float(mean_gates.max()),
                "min_mean_gate": float(mean_gates.min()),
                "likely_uniform": bool(normalized_entropy > 0.95),
                "likely_collapse": bool(mean_gates.max() > 0.95),
            }
        )
    return pd.DataFrame(rows)


def expert_cosine_stats_table(pair_cosine, labels, metadata):
    labels = np.asarray(labels).astype(int)
    num_blocks = pair_cosine.shape[1]
    pair_count = pair_cosine.shape[2]
    rows = []
    for block_index in range(num_blocks):
        expert_pairs = _expert_pairs_from_pair_count(pair_count)
        for pair_index, (left, right) in enumerate(expert_pairs[:pair_count]):
            values = pair_cosine[:, block_index, pair_index]
            pair_name = "expert{}_expert{}".format(left + 1, right + 1)
            for group_name, mask in [("normal", labels == 0), ("anomaly", labels == 1)]:
                rows.append(
                    {
                        "block": block_index + 1,
                        "pair": pair_name,
                        "left_patch_size": _patch_size(metadata, block_index, left),
                        "right_patch_size": _patch_size(metadata, block_index, right),
                        "label_group": group_name,
                        **_stats(values[mask]),
                        "min": float(np.min(values[mask])) if mask.any() else float("nan"),
                        "max": float(np.max(values[mask])) if mask.any() else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def pre_post_graph_cosine_table(router_scores, labels, metadata):
    labels = np.asarray(labels).astype(int)
    arrays = [
        ("pre_graph", router_scores.get("expert_pre_graph_pair_cosine")),
        ("post_graph", router_scores.get("expert_post_graph_pair_cosine")),
    ]
    if any(values is None for _, values in arrays):
        return pd.DataFrame()
    rows = []
    for position, pair_cosine in arrays:
        num_blocks = pair_cosine.shape[1]
        pair_count = pair_cosine.shape[2]
        for block_index in range(num_blocks):
            for pair_index, (left, right) in enumerate(
                _expert_pairs_from_pair_count(pair_count)
            ):
                values = pair_cosine[:, block_index, pair_index]
                pair_name = "expert{}_expert{}".format(left + 1, right + 1)
                for group_name, mask in [
                    ("all", np.ones(labels.shape[0], dtype=bool)),
                    ("normal", labels == 0),
                    ("anomaly", labels == 1),
                ]:
                    selected = values[mask]
                    rows.append(
                        {
                            "block": block_index + 1,
                            "position": position,
                            "expert_pair": pair_name,
                            "left_patch_size": _patch_size(
                                metadata,
                                block_index,
                                left,
                            ),
                            "right_patch_size": _patch_size(
                                metadata,
                                block_index,
                                right,
                            ),
                            "label_group": group_name,
                            **_stats(selected),
                            "min": float(np.min(selected))
                            if selected.size
                            else float("nan"),
                            "max": float(np.max(selected))
                            if selected.size
                            else float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


def graph_trace_cosine_table(router_scores, labels, metadata):
    trace = router_scores.get("graph_trace_pair_cosine")
    if trace is None:
        return pd.DataFrame()
    labels = np.asarray(labels).astype(int)
    positions = metadata.get(
        "graph_trace_positions",
        ["input", "after_adjacency", "residual_base", "after_residual", "after_norm"],
    )
    trace = np.asarray(trace)
    num_blocks = trace.shape[1]
    num_positions = trace.shape[2]
    pair_count = trace.shape[3]
    rows = []
    for block_index in range(num_blocks):
        pairs = _expert_pairs_from_pair_count(pair_count)
        for position_index in range(num_positions):
            position = (
                positions[position_index]
                if position_index < len(positions)
                else "position_{}".format(position_index + 1)
            )
            for pair_index, (left, right) in enumerate(pairs):
                values = trace[:, block_index, position_index, pair_index]
                pair_name = "expert{}_expert{}".format(left + 1, right + 1)
                for group_name, mask in [
                    ("all", np.ones(labels.shape[0], dtype=bool)),
                    ("normal", labels == 0),
                    ("anomaly", labels == 1),
                ]:
                    selected = values[mask]
                    rows.append(
                        {
                            "block": block_index + 1,
                            "position": position,
                            "expert_pair": pair_name,
                            "left_patch_size": _patch_size(
                                metadata,
                                block_index,
                                left,
                            ),
                            "right_patch_size": _patch_size(
                                metadata,
                                block_index,
                                right,
                            ),
                            "label_group": group_name,
                            **_stats(selected),
                            "min": float(np.min(selected))
                            if selected.size
                            else float("nan"),
                            "max": float(np.max(selected))
                            if selected.size
                            else float("nan"),
                        }
                    )
    return pd.DataFrame(rows)


def graph_trace_summary_table(trace_table):
    if trace_table.empty:
        return pd.DataFrame()
    table = trace_table[trace_table["label_group"] == "all"]
    positions = list(dict.fromkeys(trace_table["position"].tolist()))
    position_order = {position: index for index, position in enumerate(positions)}
    pivot = (
        table.groupby(["position", "block"])["mean"]
        .mean()
        .unstack()
        .reset_index()
        .rename(columns={"position": "Position"})
    )
    pivot["_position_order"] = pivot["Position"].map(position_order)
    pivot = pivot.sort_values("_position_order").drop(columns=["_position_order"])
    for column in list(pivot.columns):
        if isinstance(column, int):
            pivot = pivot.rename(columns={column: "B{} cosine".format(column)})
    return pivot


def single_expert_auc_table(router_scores, labels, metadata):
    rows = []
    for score_key, score_name in [
        ("single_expert_mse_topk", "single_expert_MSE"),
        ("single_expert_nll_topk", "single_expert_NLL"),
    ]:
        values = router_scores.get(score_key)
        if values is None:
            continue
        for expert_index in range(values.shape[1]):
            rows.append(
                {
                    "expert": expert_index + 1,
                    "patch_path": _expert_path(metadata, expert_index),
                    "score_name": score_name,
                    "ROC-AUC": _safe_auc(labels, values[:, expert_index], "roc"),
                    "PR-AUC": _safe_auc(labels, values[:, expert_index], "pr"),
                }
            )
    return pd.DataFrame(rows)


def expert_score_fusion_auc_table(router_scores, prob_scores, labels):
    rows = []
    score_configs = [
        ("single_expert_mse_topk", "S_MSE_topk", "MSE"),
        ("single_expert_nll_topk", "S_NLL_topk", "NLL"),
    ]
    for single_key, current_key, score_name in score_configs:
        expert_scores = router_scores.get(single_key)
        if expert_scores is None:
            continue
        expert_scores = np.asarray(expert_scores, dtype=np.float64)
        fusion_scores = {
            "E{}".format(index + 1): expert_scores[:, index]
            for index in range(expert_scores.shape[1])
        }
        fusion_scores["Mean"] = expert_scores.mean(axis=1)
        fusion_scores["Max"] = expert_scores.max(axis=1)
        current_router = prob_scores.get(current_key)
        if current_router is not None:
            fusion_scores["CurrentRouter"] = current_router
        for fusion_name, values in fusion_scores.items():
            rows.append(
                {
                    "score_name": score_name,
                    "fusion": fusion_name,
                    "ROC-AUC": _safe_auc(labels, values, "roc"),
                    "PR-AUC": _safe_auc(labels, values, "pr"),
                }
            )
    return pd.DataFrame(rows)


def single_expert_auc_by_anomaly_type_table(
    router_scores,
    labels,
    aligned_meta,
    metadata,
):
    if aligned_meta is None or "fine_anomaly_type" not in aligned_meta.columns:
        return pd.DataFrame()

    labels = np.asarray(labels).astype(int)
    anomaly_types = aligned_meta["fine_anomaly_type"].astype(str).to_numpy()
    fault_types = [
        name
        for name in sorted(set(anomaly_types), key=anomaly_type_sort_key)
        if name != "normal" and np.any((anomaly_types == name) & (labels == 1))
    ]

    rows = []
    for score_key, score_name in [
        ("single_expert_mse_topk", "MSE"),
        ("single_expert_nll_topk", "NLL"),
    ]:
        expert_scores = router_scores.get(score_key)
        if expert_scores is None:
            continue
        expert_scores = np.asarray(expert_scores, dtype=np.float64)
        for fault_type in fault_types:
            mask = anomaly_types == fault_type
            masked_labels = labels[mask]
            for expert_index in range(expert_scores.shape[1]):
                values = expert_scores[mask, expert_index]
                rows.append(
                    {
                        "anomaly_type": fault_type,
                        "score_name": score_name,
                        "expert": expert_index + 1,
                        "patch_path": _expert_path(metadata, expert_index),
                        "ROC-AUC": _safe_auc(masked_labels, values, "roc"),
                        "PR-AUC": _safe_auc(masked_labels, values, "pr"),
                        "points": int(mask.sum()),
                        "anomaly_points": int(masked_labels.sum()),
                    }
                )
    return pd.DataFrame(rows)


def adjacency_stats_table(adjacency, metadata):
    adjacency_mean = np.asarray(adjacency, dtype=np.float64).mean(axis=0)
    rows = []
    for block_index in range(adjacency_mean.shape[0]):
        for expert_index in range(adjacency_mean.shape[1]):
            matrix = adjacency_mean[block_index, expert_index]
            offdiag_mask = ~np.eye(matrix.shape[0], dtype=bool)
            offdiag = matrix[offdiag_mask]
            rows.append(
                {
                    "block": block_index + 1,
                    "expert": expert_index + 1,
                    "patch_size": _patch_size(metadata, block_index, expert_index),
                    "mean_weight": float(matrix.mean()),
                    "std_weight": float(matrix.std()),
                    "offdiag_mean": float(offdiag.mean()),
                    "offdiag_std": float(offdiag.std()),
                    "offdiag_nonzero_ratio": float(np.mean(np.abs(offdiag) > 1e-12)),
                    "diag_mean": float(np.diag(matrix).mean()),
                    "min": float(matrix.min()),
                    "max": float(matrix.max()),
                }
            )
    return pd.DataFrame(rows)


def obs_feature_auc_table(obs_feature, labels, feature_names):
    if obs_feature is None:
        return pd.DataFrame()
    rows = []
    for index, name in enumerate(feature_names):
        values = obs_feature[:, index]
        rows.append(
            {
                "feature": name,
                "ROC-AUC": _safe_auc(labels, values, "roc"),
                "PR-AUC": _safe_auc(labels, values, "pr"),
            }
        )
    return pd.DataFrame(rows)


def m2_prediction_feature_auc_table(prediction_feature_errors, labels, feature_names):
    if prediction_feature_errors is None:
        return pd.DataFrame()
    rows = []
    for index, feature_name in enumerate(feature_names):
        values = prediction_feature_errors[:, index]
        rows.append(
            {
                "feature": feature_name,
                "ROC-AUC": _safe_auc(labels, values, "roc"),
                "PR-AUC": _safe_auc(labels, values, "pr"),
            }
        )
    return pd.DataFrame(rows)


def m2_train_standardized_topk_auc_table(
    prediction_feature_errors,
    labels,
    feature_scale,
    feature_scale_source,
    topk=3,
):
    if prediction_feature_errors is None or feature_scale is None:
        return pd.DataFrame()
    standardized_topk = aggregate_feature_errors(
        prediction_feature_errors,
        strategy="train_feature_topk",
        topk=topk,
        feature_scale=feature_scale,
    )
    return pd.DataFrame(
        [
            {
                "score_name": "M2_train_standardized_topk",
                "topk": int(max(1, min(int(topk), prediction_feature_errors.shape[1]))),
                "feature_scale_source": feature_scale_source,
                "ROC-AUC": _safe_auc(labels, standardized_topk, "roc"),
                "PR-AUC": _safe_auc(labels, standardized_topk, "pr"),
            }
        ]
    )


def obs_feature_stats_table(obs_feature, labels, feature_names):
    if obs_feature is None:
        return pd.DataFrame()
    labels = np.asarray(labels).astype(int)
    rows = []
    for index, name in enumerate(feature_names):
        values = obs_feature[:, index]
        for group_name, mask in [("normal", labels == 0), ("anomaly", labels == 1)]:
            rows.append(
                {
                    "feature": name,
                    "label_group": group_name,
                    **_stats(values[mask]),
                }
            )
    return pd.DataFrame(rows)


def probabilistic_feature_auc_table(feature_scores, labels, feature_names):
    rows = []
    for score_name, values in feature_scores.items():
        for index, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "score_name": score_name,
                    "feature": feature_name,
                    "ROC-AUC": _safe_auc(labels, values[:, index], "roc"),
                    "PR-AUC": _safe_auc(labels, values[:, index], "pr"),
                }
            )
    return pd.DataFrame(rows)


def logvar_stats_table(logvar_feature, labels, feature_names, lower=-6.0, upper=4.0):
    labels = np.asarray(labels).astype(int)
    rows = []

    def add_row(scope, group_name, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        row = {
            "scope": scope,
            "label_group": group_name,
            "count": int(values.size),
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "median": float(np.median(values)) if values.size else float("nan"),
            "std": float(np.std(values)) if values.size else float("nan"),
            "min": float(np.min(values)) if values.size else float("nan"),
            "max": float(np.max(values)) if values.size else float("nan"),
            "P5": float(np.percentile(values, 5)) if values.size else float("nan"),
            "P95": float(np.percentile(values, 95)) if values.size else float("nan"),
            "lower_clamp_ratio": float(np.mean(values <= lower + 1e-6))
            if values.size
            else float("nan"),
            "upper_clamp_ratio": float(np.mean(values >= upper - 1e-6))
            if values.size
            else float("nan"),
        }
        rows.append(row)

    for group_name, mask in [("normal", labels == 0), ("anomaly", labels == 1)]:
        add_row("all_features", group_name, logvar_feature[mask])
        for index, feature_name in enumerate(feature_names):
            add_row(feature_name, group_name, logvar_feature[mask, index])

    return pd.DataFrame(rows)


def plot_score_distributions(scores, labels, output_path):
    labels = np.asarray(labels).astype(int)
    fig, axes = plt.subplots(1, len(scores), figsize=(5 * len(scores), 4))
    if len(scores) == 1:
        axes = [axes]
    for axis, (name, values) in zip(axes, scores.items()):
        values = np.asarray(values, dtype=np.float64)
        axis.hist(
            values[labels == 0],
            bins=50,
            alpha=0.65,
            density=True,
            label="normal",
            color="#3b82f6",
        )
        axis.hist(
            values[labels == 1],
            bins=50,
            alpha=0.65,
            density=True,
            label="anomaly",
            color="#ef4444",
        )
        axis.set_title(name)
        axis.set_xlabel("score")
        axis.set_ylabel("density")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_logvar_distribution(logvar_feature, labels, output_path):
    labels = np.asarray(labels).astype(int)
    normal = logvar_feature[labels == 0].reshape(-1)
    anomaly = logvar_feature[labels == 1].reshape(-1)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.hist(normal, bins=60, alpha=0.65, density=True, label="normal", color="#3b82f6")
    axis.hist(anomaly, bins=60, alpha=0.65, density=True, label="anomaly", color="#ef4444")
    axis.axvline(-6.0, linestyle="--", linewidth=1.1, color="#111827", label="lower clamp")
    axis.axvline(4.0, linestyle=":", linewidth=1.1, color="#111827", label="upper clamp")
    axis.set_title("reconstruction logvar")
    axis.set_xlabel("logvar")
    axis.set_ylabel("density")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_router_gate_distribution(router_gates, labels, metadata, output_path):
    labels = np.asarray(labels).astype(int)
    num_blocks = router_gates.shape[1]
    num_experts = router_gates.shape[2]
    fig, axes = plt.subplots(num_blocks, 1, figsize=(10, 3.2 * num_blocks), squeeze=False)
    for block_index in range(num_blocks):
        axis = axes[block_index, 0]
        x = np.arange(num_experts)
        normal_mean = router_gates[labels == 0, block_index].mean(axis=0)
        anomaly_mean = router_gates[labels == 1, block_index].mean(axis=0)
        width = 0.36
        axis.bar(x - width / 2, normal_mean, width=width, label="normal", color="#3b82f6")
        axis.bar(x + width / 2, anomaly_mean, width=width, label="anomaly", color="#ef4444")
        labels_text = [
            "E{} p{}".format(expert_index + 1, _patch_size(metadata, block_index, expert_index))
            for expert_index in range(num_experts)
        ]
        axis.set_xticks(x)
        axis.set_xticklabels(labels_text)
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("mean gate")
        axis.set_title("Router gates - block {}".format(block_index + 1))
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_adjacency_heatmaps(adjacency, metadata, output_dir):
    adjacency_mean = np.asarray(adjacency, dtype=np.float64).mean(axis=0)
    created = []
    for block_index in range(adjacency_mean.shape[0]):
        for expert_index in range(adjacency_mean.shape[1]):
            patch_size = _patch_size(metadata, block_index, expert_index)
            matrix = adjacency_mean[block_index, expert_index]
            fig, axis = plt.subplots(figsize=(5, 4.5))
            image = axis.imshow(matrix, cmap="viridis", aspect="auto")
            axis.set_title(
                "Adjacency block {} expert {} patch {}".format(
                    block_index + 1,
                    expert_index + 1,
                    patch_size,
                )
            )
            axis.set_xlabel("source variable")
            axis.set_ylabel("target variable")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            fig.tight_layout()
            output_path = output_dir / (
                "r4_adjacency_block{}_expert{}_patch{}.png".format(
                    block_index + 1,
                    expert_index + 1,
                    patch_size,
                )
            )
            fig.savefig(output_path, dpi=160)
            plt.close(fig)
            created.append(output_path)
    return created


def _event_slice_for_fault(aligned_meta, labels, fault_type):
    if aligned_meta is None:
        return None
    fault_mask = aligned_meta["fine_anomaly_type"].astype(str).to_numpy() == fault_type
    anomaly_mask = fault_mask & (np.asarray(labels).astype(int) == 1)
    if not anomaly_mask.any():
        return None

    anomaly_positions = np.flatnonzero(anomaly_mask)
    start = int(anomaly_positions[0])
    end = int(anomaly_positions[-1]) + 1
    if "segment_id" in aligned_meta.columns:
        segment_id = aligned_meta.iloc[start]["segment_id"]
        segment_positions = np.flatnonzero(
            aligned_meta["segment_id"].to_numpy() == segment_id
        )
        start = max(int(segment_positions[0]), start - 200)
        end = min(int(segment_positions[-1]) + 1, end + 200)
    else:
        start = max(0, start - 200)
        end = min(len(labels), end + 200)
    return slice(start, end)


def plot_fault_timeseries(scores, labels, threshold, aligned_meta, fault_types, output_dir):
    created = []
    x_all = np.arange(len(labels))
    threshold = np.asarray(threshold, dtype=np.float64).reshape(-1)
    if threshold.size == 1:
        threshold = np.full(len(labels), float(threshold[0]))

    for fault_type in fault_types:
        event_slice = _event_slice_for_fault(aligned_meta, labels, fault_type)
        if event_slice is None:
            continue

        x = x_all[event_slice]
        fig, axis = plt.subplots(figsize=(12, 4.8))
        if "S_obs_topk" in scores:
            axis.plot(
                x,
                scores["S_obs_topk"][event_slice],
                label="S_obs",
                linewidth=1.4,
            )
        if "S_dyn" in scores:
            axis.plot(
                x,
                scores["S_dyn"][event_slice],
                label="S_dyn",
                linewidth=1.2,
            )
        axis.plot(x, scores["S_total"][event_slice], label="S_total", linewidth=1.4)
        axis.plot(
            x,
            threshold[event_slice],
            label="threshold",
            linewidth=1.2,
            linestyle="--",
            color="#111827",
        )

        local_labels = np.asarray(labels)[event_slice].astype(int)
        if local_labels.any():
            ymin, ymax = axis.get_ylim()
            axis.fill_between(
                x,
                ymin,
                ymax,
                where=local_labels == 1,
                color="#ef4444",
                alpha=0.12,
                label="GT anomaly",
            )
            axis.set_ylim(ymin, ymax)

        axis.set_title(fault_type)
        axis.set_xlabel("scored point index")
        axis.set_ylabel("score")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=3)
        fig.tight_layout()
        output_path = output_dir / ("s2_timeseries_" + _slug(fault_type) + ".png")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        created.append(output_path)
    return created


def plot_mse_nll_timeseries(prob_scores, labels, aligned_meta, fault_types, output_dir):
    created = []
    if not prob_scores:
        return created
    x_all = np.arange(len(labels))
    for fault_type in fault_types:
        event_slice = _event_slice_for_fault(aligned_meta, labels, fault_type)
        if event_slice is None:
            continue

        x = x_all[event_slice]
        fig, axis = plt.subplots(figsize=(12, 4.8))
        axis.plot(
            x,
            prob_scores["S_MSE_topk"][event_slice],
            label="MSE",
            linewidth=1.4,
            color="#2563eb",
        )
        axis.plot(
            x,
            prob_scores["S_NLL_topk"][event_slice],
            label="NLL",
            linewidth=1.4,
            color="#16a34a",
        )
        if "S_stdres_topk" in prob_scores:
            axis.plot(
                x,
                prob_scores["S_stdres_topk"][event_slice],
                label="standardized residual",
                linewidth=1.0,
                alpha=0.75,
                color="#f97316",
            )

        local_labels = np.asarray(labels)[event_slice].astype(int)
        if local_labels.any():
            ymin, ymax = axis.get_ylim()
            axis.fill_between(
                x,
                ymin,
                ymax,
                where=local_labels == 1,
                color="#ef4444",
                alpha=0.12,
                label="GT anomaly",
            )
            axis.set_ylim(ymin, ymax)

        axis.set_title("MSE vs NLL - " + fault_type)
        axis.set_xlabel("scored point index")
        axis.set_ylabel("score")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=3)
        fig.tight_layout()
        output_path = output_dir / ("s6_mse_vs_nll_" + _slug(fault_type) + ".png")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        created.append(output_path)
    return created


def run_score_diagnostics(
    result_dir,
    data_root=None,
    test_meta_path=None,
    output_dir=None,
    feature_names=None,
    fault_types=None,
    obs_topk=3,
):
    result_dir = Path(result_dir)
    data_root = Path(data_root) if data_root is not None else None
    if test_meta_path is None and data_root is not None:
        test_meta_path = data_root / "test_meta.csv"
    output_dir = Path(output_dir) if output_dir is not None else result_dir / "score_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores, obs_feature, labels, indices, threshold = load_saved_scores(result_dir)
    prob_scores, prob_feature_scores = load_probabilistic_scores(result_dir)
    prediction_feature_errors = load_prediction_feature_errors(result_dir)
    train_feature_scale, train_feature_scale_source = load_train_feature_error_scale(
        result_dir
    )
    router_scores = load_router_scores(result_dir)
    router_metadata = load_router_metadata(result_dir)
    feature_names = load_feature_names(data_root, obs_feature, feature_names)
    aligned_meta = load_aligned_meta(test_meta_path, indices) if test_meta_path else None

    component_auc_table(scores, labels).to_csv(
        output_dir / "s1_component_auc.csv",
        index=False,
    )
    distribution_stats_table(scores, labels).to_csv(
        output_dir / "s1_component_distribution_stats.csv",
        index=False,
    )
    type_auc_table = score_auc_by_anomaly_type_table(scores, labels, aligned_meta)
    if not type_auc_table.empty:
        type_auc_table.to_csv(
            output_dir / "s1_score_auc_by_anomaly_type.csv",
            index=False,
        )
        plot_score_auc_by_anomaly_type(
            type_auc_table,
            output_dir / "s1_score_auc_by_anomaly_type.png",
        )
    phase_table = event_phase_score_stats_table(scores, labels, aligned_meta)
    if not phase_table.empty:
        phase_table.to_csv(
            output_dir / "a6_event_phase_score_stats.csv",
            index=False,
        )
    obs_feature_auc_table(obs_feature, labels, feature_names).to_csv(
        output_dir / "s1_obs_feature_auc.csv",
        index=False,
    )
    obs_feature_stats_table(obs_feature, labels, feature_names).to_csv(
        output_dir / "s1_obs_feature_distribution_stats.csv",
        index=False,
    )
    m2_prediction_feature_auc = m2_prediction_feature_auc_table(
        prediction_feature_errors,
        labels,
        feature_names,
    )
    if not m2_prediction_feature_auc.empty:
        m2_prediction_feature_auc.to_csv(
            output_dir / "m2_prediction_feature_auc.csv",
            index=False,
        )
    m2_standardized_topk_auc = m2_train_standardized_topk_auc_table(
        prediction_feature_errors,
        labels,
        train_feature_scale,
        train_feature_scale_source,
        topk=obs_topk,
    )
    if not m2_standardized_topk_auc.empty:
        m2_standardized_topk_auc.to_csv(
            output_dir / "m2_train_standardized_topk_auc.csv",
            index=False,
        )
    plot_score_distributions(
        scores,
        labels,
        output_dir / "s3_score_distribution.png",
    )
    plot_fault_timeseries(
        scores,
        labels,
        threshold,
        aligned_meta,
        fault_types or DEFAULT_FAULT_TYPES,
        output_dir,
    )
    if prob_scores and prob_feature_scores:
        component_auc_table(prob_scores, labels).to_csv(
            output_dir / "s4_probabilistic_score_auc.csv",
            index=False,
        )
        distribution_stats_table(prob_scores, labels).to_csv(
            output_dir / "s4_probabilistic_score_distribution_stats.csv",
            index=False,
        )
        probabilistic_feature_auc_table(
            prob_feature_scores,
            labels,
            feature_names,
        ).to_csv(
            output_dir / "s4_probabilistic_feature_auc.csv",
            index=False,
        )
        probabilistic_feature_auc_table(
            {"S_unc_logvar": prob_feature_scores["S_unc_logvar"]},
            labels,
            feature_names,
        ).to_csv(
            output_dir / "s5_logvar_feature_auc.csv",
            index=False,
        )
        logvar_stats = logvar_stats_table(
            prob_feature_scores["S_unc_logvar"],
            labels,
            feature_names,
        )
        logvar_stats.to_csv(output_dir / "s5_logvar_stats.csv", index=False)
        plot_logvar_distribution(
            prob_feature_scores["S_unc_logvar"],
            labels,
            output_dir / "s5_logvar_distribution.png",
        )
        plot_mse_nll_timeseries(
            prob_scores,
            labels,
            aligned_meta,
            fault_types or DEFAULT_FAULT_TYPES,
            output_dir,
        )
    router_gates = router_scores.get("router_gates")
    if router_gates is not None:
        router_gate_stats_table(router_gates, labels, router_metadata).to_csv(
            output_dir / "r1_router_gate_stats.csv",
            index=False,
        )
        router_entropy_stats_table(router_gates, labels).to_csv(
            output_dir / "r1_router_entropy_stats.csv",
            index=False,
        )
        router_summary_table(router_gates).to_csv(
            output_dir / "r1_router_summary.csv",
            index=False,
        )
        plot_router_gate_distribution(
            router_gates,
            labels,
            router_metadata,
            output_dir / "r1_router_gate_distribution.png",
        )
    pair_cosine = router_scores.get("expert_pair_cosine")
    if pair_cosine is not None:
        expert_cosine_stats_table(pair_cosine, labels, router_metadata).to_csv(
            output_dir / "r2_expert_cosine_stats.csv",
            index=False,
        )
    g1_table = pre_post_graph_cosine_table(router_scores, labels, router_metadata)
    if not g1_table.empty:
        g1_table.to_csv(
            output_dir / "g1_pre_post_graph_cosine.csv",
            index=False,
        )
    g4_table = graph_trace_cosine_table(router_scores, labels, router_metadata)
    if not g4_table.empty:
        g4_table.to_csv(
            output_dir / "g4_graph_trace_cosine.csv",
            index=False,
        )
        graph_trace_summary_table(g4_table).to_csv(
            output_dir / "g4_graph_trace_summary.csv",
            index=False,
        )
    if (
        router_scores.get("single_expert_mse_topk") is not None
        or router_scores.get("single_expert_nll_topk") is not None
    ):
        single_expert_auc_table(router_scores, labels, router_metadata).to_csv(
            output_dir / "r3_single_expert_auc.csv",
            index=False,
        )
        expert_score_fusion_auc_table(router_scores, prob_scores, labels).to_csv(
            output_dir / "r5_expert_score_fusion_auc.csv",
            index=False,
        )
        single_expert_auc_by_anomaly_type_table(
            router_scores,
            labels,
            aligned_meta,
            router_metadata,
        ).to_csv(
            output_dir / "r6_single_expert_auc_by_anomaly_type.csv",
            index=False,
        )
    adjacency = router_scores.get("adjacency")
    if adjacency is not None:
        adjacency_stats_table(adjacency, router_metadata).to_csv(
            output_dir / "r4_adjacency_stats.csv",
            index=False,
        )
        plot_adjacency_heatmaps(adjacency, router_metadata, output_dir)
    return output_dir
