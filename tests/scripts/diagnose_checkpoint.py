import argparse
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from data_provider.data_loader import Dataset_ALFA
from models.MSTGCNet import Model
from run import build_parser, normalize_args


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose an MSTGCNet checkpoint.")
    parser.add_argument("--setting", required=True)
    parser.add_argument("--sample-size", type=int, default=256)
    return parser.parse_args()


def select_starts(data, sample_size, seed=2025):
    rng = np.random.default_rng(seed)
    train_starts = np.asarray(data.train_windows)
    test_starts = np.asarray(data.test_windows)
    test_end_indices = test_starts + data.win_size - 1
    test_end_labels = data.test_labels[test_end_indices].astype(int)
    normal_starts = test_starts[test_end_labels == 0]
    anomaly_starts = test_starts[test_end_labels == 1]

    def sample(values):
        size = min(sample_size, len(values))
        return rng.choice(values, size=size, replace=False)

    return sample(train_starts), sample(normal_starts), sample(anomaly_starts)


def infer(model, values, marks, starts, batch_size=64):
    inputs = []
    outputs = []
    balance_losses = []
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset: offset + batch_size]
            batch_x = torch.from_numpy(
                np.stack([values[start:start + model.seq_len] for start in batch_starts])
            ).float()
            batch_m = torch.from_numpy(
                np.stack([marks[start:start + model.seq_len] for start in batch_starts])
            ).float()
            output, balance_loss = model(batch_x, batch_m)
            inputs.append(batch_x)
            outputs.append(output)
            balance_losses.append(float(balance_loss))
    return torch.cat(inputs), torch.cat(outputs), float(np.mean(balance_losses))


def describe(name, inputs, outputs, balance_loss):
    error = (inputs - outputs).pow(2)
    full_mse = float(error.mean())
    reconstruction_loss = float(error.flatten(start_dim=1).sum(dim=1).mean())
    endpoint_error = error[:, -1]
    correlation = torch.corrcoef(
        torch.stack([inputs.flatten(), outputs.flatten()])
    )[0, 1]
    print(
        name,
        "full_mse=", full_mse,
        "reconstruction_loss=", reconstruction_loss,
        "last_mse=", float(endpoint_error.mean()),
        "balance=", balance_loss,
        "balance/reconstruction=",
        balance_loss / max(reconstruction_loss, 1e-12),
    )
    print(
        name,
        "input_std=", float(inputs.std()),
        "output_std=", float(outputs.std()),
        "correlation=", float(correlation),
    )
    print(name, "per_variable_last_mse=", endpoint_error.mean(0).numpy().tolist())
    return endpoint_error.mean(-1).numpy(), endpoint_error.numpy()


def routing_stats(model, inputs):
    with torch.no_grad():
        hidden = model.revin_layer(inputs, "norm") if model.revin else inputs
        hidden = model.conv_embedding(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = model.conv_scale * hidden + model.position_embedding(hidden)
        result = []
        for block in model.blocks:
            sparse_weights, _, _ = block.router(hidden)
            selected = torch.topk(
                sparse_weights, block.top_k, dim=-1
            ).indices
            usage = torch.bincount(
                selected.flatten(), minlength=block.num_experts
            )
            normalized = sparse_weights / sparse_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(-1).mean()
            result.append(
                {
                    "usage": usage.numpy().tolist(),
                    "entropy": float(entropy),
                    "mean_weights": sparse_weights.mean(0).numpy().tolist(),
                    "mean_selected_weight_sum": float(
                        sparse_weights.sum(dim=-1).mean()
                    ),
                }
            )
            hidden, _, _ = block(hidden)
    return result


def router_input_stats(model, inputs):
    with torch.no_grad():
        hidden = model.revin_layer(inputs, "norm") if model.revin else inputs
        hidden = model.conv_embedding(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = model.conv_scale * hidden + model.position_embedding(hidden)
        router = model.blocks[0].router
        transformed = router.merge(
            hidden + router._seasonal(hidden) + router._trend(hidden)
        )
        routing_input = router.channel_projection(transformed).squeeze(-1)
        sparse_weights, _, _ = router(hidden)
    return {
        "transformed_between_sample_std": float(transformed.std(dim=0).mean()),
        "routing_between_sample_std": float(routing_input.std(dim=0).mean()),
        "routing_absolute_mean": float(routing_input.abs().mean()),
        "weight_between_sample_std": float(
            sparse_weights.std(dim=0).mean()
        ),
    }


def balance_gradient_stats(model, inputs, marks):
    model.train()
    model.zero_grad(set_to_none=True)
    _, balance_loss = model(inputs[:16], marks[:16])
    result = {
        "value": float(balance_loss.detach()),
        "requires_grad": bool(balance_loss.requires_grad),
        "grad_fn": type(balance_loss.grad_fn).__name__
        if balance_loss.grad_fn is not None
        else None,
    }
    model.eval()
    return result


def residual_path_stats(model, inputs):
    with torch.no_grad():
        normalized = model.revin_layer(inputs, "norm") if model.revin else inputs
        hidden = model.conv_embedding(normalized.transpose(1, 2)).transpose(1, 2)
        hidden = model.conv_scale * hidden + model.position_embedding(hidden)
        block_ratios = []
        for block in model.blocks:
            next_hidden, _, _ = block(hidden)
            block_ratios.append(
                float((next_hidden - hidden).norm() / hidden.norm().clamp_min(1e-12))
            )
            hidden = next_hidden

        bypass_hidden = model.conv_embedding(
            normalized.transpose(1, 2)
        ).transpose(1, 2)
        bypass_hidden = (
            model.conv_scale * bypass_hidden
            + model.position_embedding(bypass_hidden)
        )
        bypass_output = model.reconstruction(bypass_hidden)
        if model.revin:
            bypass_output = model.revin_layer(bypass_output, "denorm")
        bypass_error = (inputs - bypass_output).pow(2)
    return {
        "block_delta_ratios": block_ratios,
        "bypass_full_mse": float(bypass_error.mean()),
        "bypass_last_mse": float(bypass_error[:, -1].mean()),
    }


def main():
    cli = parse_args()
    args = normalize_args(build_parser().parse_args(["--use_gpu", "false"]))
    checkpoint = os.path.join("checkpoints", cli.setting, "checkpoint.pth")

    model = Model(args)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    data = Dataset_ALFA(args, args.root_path, args.seq_len, flag="test")
    train_starts, normal_starts, anomaly_starts = select_starts(
        data, cli.sample_size
    )

    started = time.time()
    train_x, train_out, train_balance = infer(
        model, data.train, data.train_mark, train_starts
    )
    normal_x, normal_out, normal_balance = infer(
        model, data.test, data.test_mark, normal_starts
    )
    anomaly_x, anomaly_out, anomaly_balance = infer(
        model, data.test, data.test_mark, anomaly_starts
    )

    _, _ = describe("train", train_x, train_out, train_balance)
    normal_score, normal_variables = describe(
        "test_normal", normal_x, normal_out, normal_balance
    )
    anomaly_score, anomaly_variables = describe(
        "test_anomaly", anomaly_x, anomaly_out, anomaly_balance
    )

    labels = np.r_[np.zeros(len(normal_score)), np.ones(len(anomaly_score))]
    print(
        "sample_endpoint_auc=",
        roc_auc_score(labels, np.r_[normal_score, anomaly_score]),
    )
    for variable in range(normal_variables.shape[1]):
        variable_score = np.r_[
            normal_variables[:, variable], anomaly_variables[:, variable]
        ]
        print("sample_variable_auc", variable, roc_auc_score(labels, variable_score))

    for name, inputs in (
        ("train", train_x),
        ("test_normal", normal_x),
        ("test_anomaly", anomaly_x),
    ):
        print("routing", name, routing_stats(model, inputs))
    print("router_input", router_input_stats(model, train_x))
    print("residual_path_train", residual_path_stats(model, train_x))
    print("residual_path_normal", residual_path_stats(model, normal_x))
    print("residual_path_anomaly", residual_path_stats(model, anomaly_x))
    train_marks = torch.from_numpy(
        np.stack(
            [
                data.train_mark[start:start + model.seq_len]
                for start in train_starts[:16]
            ]
        )
    ).float()
    print("balance_gradient", balance_gradient_stats(model, train_x, train_marks))
    print("conv_scale=", float(model.conv_scale))
    print("elapsed_seconds=", time.time() - started)


if __name__ == "__main__":
    main()
