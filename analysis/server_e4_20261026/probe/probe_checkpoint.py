"""Bounded CPU probe of the existing E4 checkpoint; run from the project root.

    python -B analysis/server_e4_20261026/probe/probe_checkpoint.py

Writes only results.json beside this script. No training or full-set evaluation.
"""
import sys

sys.dont_write_bytecode = True

import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "analysis/server_e4_20261026/probe/results.json"
sys.path.insert(0, str(ROOT))

from data_provider.data_loader import Dataset_ALFA
from models.MSTGCNet import Model


def comparison(reference, changed):
    difference = np.abs(reference - changed)
    return {
        "relative_l1": float(difference.sum() / np.abs(reference).sum()),
        "mean_absolute": float(difference.mean()),
        "max_absolute": float(difference.max()),
        "pearson_correlation": float(np.corrcoef(reference.ravel(), changed.ravel())[0, 1]),
    }


def main():
    torch.set_num_threads(2)
    checkpoint_dirs = sorted((ROOT / "服务器训练结果/checkpoints").glob("*_E4_*"))
    if len(checkpoint_dirs) != 1:
        raise RuntimeError("Expected exactly one server E4 checkpoint directory")
    checkpoint_dir = checkpoint_dirs[0]
    checkpoint_path = checkpoint_dir / "checkpoint.pth"
    result_dir = ROOT / "服务器训练结果/label_results" / checkpoint_dir.name / "causal_last"
    config = json.loads((checkpoint_dir / "experiment_config.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(**config["args"])
    if args.experiment_mode != "E4" or args.lambda_prior != 1 or args.score_normalization != "none":
        raise RuntimeError("This probe requires the recorded E4, lambda=1, unnormalized scores")
    args.device = torch.device("cpu")
    args.use_gpu = False
    data = Dataset_ALFA(args, str(ROOT / "dataset/ALFA10vars_20hz"), args.seq_len, flag="test")
    model = Model(args).float()
    state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    starts = np.asarray(data.test_windows)
    endpoints = starts + args.seq_len - 1
    segments = data.test_meta.segment_id.to_numpy()
    chosen = []
    for segment in np.unique(segments):
        for anomalous in (False, True):
            candidates = np.flatnonzero(
                (segments[endpoints] == segment)
                & ((data.test_labels[endpoints] != 0) == anomalous)
            )
            if len(candidates) < 3:
                raise RuntimeError("Every segment must have at least 3 endpoints in each label class")
            chosen.extend(candidates[np.linspace(0, len(candidates) - 1, 3).astype(int)].tolist())
    selected_starts = starts[chosen]
    selected_endpoints = endpoints[chosen]
    if len(chosen) != 42:
        raise RuntimeError("Expected seven segments and exactly 42 selected windows")

    router_gates = {1: [[], [], []], 0: [[], [], []]}
    context = {"lambda": 1, "endpoints": [], "gates": {}}
    hooks = []
    graph_comparisons = []
    captured = set()
    for layer, block in enumerate(model.blocks):
        def router_hook(module, inputs, output, layer=layer):
            gates = output[0].detach().clone()
            router_gates[context["lambda"]][layer].append(gates)
            context["gates"][layer] = gates
        hooks.append(block.router.register_forward_hook(router_hook))
        for expert_index, expert in enumerate(block.experts):
            def expert_hook(module, inputs, layer=layer, expert_index=expert_index):
                if context["lambda"] != 1 or (layer, expert_index) in captured:
                    return
                captured.add((layer, expert_index))
                with torch.no_grad():
                    nodes = module._temporal_encode(module._patch(inputs[0][:1]))
                    adjacency_one = module._adjacency(nodes)[0]
                    previous_lambda = module.lambda_prior
                    try:
                        module.lambda_prior = 0.0
                        adjacency_zero = module._adjacency(nodes)[0]
                    finally:
                        module.lambda_prior = previous_lambda
                    neighbors_one = adjacency_one > 0
                    neighbors_zero = adjacency_zero > 0
                    diagonal = torch.eye(len(adjacency_one), dtype=torch.bool)
                    neighbors_one[diagonal] = False
                    neighbors_zero[diagonal] = False
                    row_l1 = (adjacency_one - adjacency_zero).abs().sum(-1)
                    changed = (neighbors_one != neighbors_zero).any(-1)
                    replaced = (neighbors_one & ~neighbors_zero).sum(-1)
                    selected = torch.nonzero(context["gates"][layer][:, expert_index] > 0, as_tuple=False).flatten()
                    graph_comparisons.append({
                        "layer": layer, "expert": expert_index, "patch_size": module.patch_size,
                        "endpoint": int(context["endpoints"][int(selected[0])]),
                        "num_nodes": len(adjacency_one),
                        "same_encoded_nodes_for_both_lambdas": True,
                        "mean_row_l1": float(row_l1.mean()), "max_row_l1": float(row_l1.max()),
                        "changed_neighbor_row_fraction": float(changed.float().mean()),
                        "changed_neighbor_rows": torch.nonzero(changed, as_tuple=False).flatten().tolist(),
                        "replaced_nonself_edges": int(replaced.sum()),
                        "nonself_edges_lambda_one": int(neighbors_one.sum()),
                    })
            hooks.append(expert.register_forward_pre_hook(expert_hook))

    reconstructions = {}
    scores = {}
    try:
        with torch.inference_mode():
            for prior_lambda in (1, 0):
                context["lambda"] = prior_lambda
                for block in model.blocks:
                    for expert in block.experts:
                        expert.lambda_prior = float(prior_lambda)
                outputs = []
                for offset in range(0, len(selected_starts), 6):
                    batch_starts = selected_starts[offset:offset + 6]
                    context["endpoints"] = selected_endpoints[offset:offset + 6]
                    inputs = torch.from_numpy(np.stack([
                        data.test[start:start + args.seq_len] for start in batch_starts
                    ])).float()
                    output, _ = model(inputs)
                    outputs.append(output.clone())
                reconstructions[prior_lambda] = torch.cat(outputs).numpy()
                targets = np.stack([data.test[start:start + args.seq_len] for start in selected_starts])
                scores[prior_lambda] = ((reconstructions[prior_lambda][:, -1] - targets[:, -1]) ** 2).mean(-1)
    finally:
        for hook in hooks:
            hook.remove()
        for block in model.blocks:
            for expert in block.experts:
                expert.lambda_prior = 1.0

    saved_indices = np.load(result_dir / "test_indices.npy", allow_pickle=False)
    saved_scores = np.load(result_dir / "test_energy.npy", allow_pickle=False)
    lookup = {int(index): position for position, index in enumerate(saved_indices)}
    reference = saved_scores[[lookup[int(index)] for index in selected_endpoints]]
    reproduction = comparison(reference, scores[1])
    reproduction["max_pointwise_relative_error"] = float(
        (np.abs(scores[1] - reference) / np.maximum(np.abs(reference), 1e-12)).max()
    )
    reproduction["allclose_rtol_1e_5_atol_1e_7"] = bool(np.allclose(scores[1], reference, rtol=1e-5, atol=1e-7))
    if not reproduction["allclose_rtol_1e_5_atol_1e_7"]:
        raise RuntimeError("Sample energy reproduction failed; do not interpret the sensitivity probe")

    router_results = {}
    for prior_lambda in (1, 0):
        layers = []
        for layer in range(len(model.blocks)):
            gates = torch.cat(router_gates[prior_lambda][layer])
            layers.append({
                "layer": layer,
                "mean_max_gate": float(gates.max(1).values.mean()),
                "mean_entropy_nats": float((-(gates * gates.clamp_min(1e-12).log()).sum(1)).mean()),
                "mean_gate_by_expert": gates.mean(0).tolist(),
                "selection_rate_by_expert": (gates > 0).float().mean(0).tolist(),
                "gates_in_endpoint_order": gates.tolist(),
            })
        router_results[str(prior_lambda)] = layers

    payload = {
        "scope": {
            "kind": "Sampled frozen-E4-checkpoint CPU sensitivity; NOT E2 retraining",
            "sampling": "Within each of 7 test segments choose first/middle/last valid normal and anomalous endpoint; 42 windows",
            "changed": "Only all expert.lambda_prior temporarily changed from 1 to 0 in memory",
            "score": "Final-point mean squared error across 10 training-standardized features; no score normalization",
            "relative_l1_definition": "sum(abs(lambda_one-lambda_zero))/sum(abs(lambda_one))",
            "graph_probe": "First routed sample per available expert; identical temporal-encoded nodes for lambda 1 and 0",
            "limits": "No training, full-set evaluation, threshold recalibration, E2 comparison, or estimate of full-test F1/AUC; cannot establish training-time value of prior",
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "device": "cpu", "threads": torch.get_num_threads(), "batch_size": 6, "eval": True},
        "checkpoint": {"path": str(checkpoint_path), "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "strict_load_all_keys_matched": True},
        "samples": [{"start": int(start), "endpoint": int(endpoint), "segment_id": int(segments[endpoint]), "label": int(data.test_labels[endpoint]), "reference_energy": float(reference[i]), "energy_lambda_one": float(scores[1][i]), "energy_lambda_zero": float(scores[0][i])} for i, (start, endpoint) in enumerate(zip(selected_starts, selected_endpoints))],
        "lambda_one_reproduction": reproduction,
        "frozen_weight_sensitivity": {
            "scores": comparison(scores[1], scores[0]),
            "final_point_reconstruction": comparison(reconstructions[1][:, -1], reconstructions[0][:, -1]),
            "all_window_reconstruction": comparison(reconstructions[1], reconstructions[0]),
            "scores_by_label": {str(label): comparison(scores[1][(data.test_labels[selected_endpoints] != 0) == bool(label)], scores[0][(data.test_labels[selected_endpoints] != 0) == bool(label)]) for label in (0, 1)},
            "scores_by_segment": {str(int(segment)): comparison(scores[1][segments[selected_endpoints] == segment], scores[0][segments[selected_endpoints] == segment]) for segment in np.unique(segments)},
        },
        "router": router_results,
        "fixed_node_adjacency_comparisons": sorted(graph_comparisons, key=lambda row: (row["layer"], row["expert"])),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sample_count": len(payload["samples"]), "reproduction": reproduction, "score_sensitivity": payload["frozen_weight_sensitivity"]["scores"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
