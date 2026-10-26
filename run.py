import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.print_args import print_args


REPRO_PATCH_SIZE_LIST = [
    [8, 12, 16, 32],
    [6, 8, 12, 16],
    [2, 6, 8, 12],
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="MSTGCNet anomaly detection")

    parser.add_argument("--task_name", type=str, default="anomaly_detection")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="ALFA10vars")
    parser.add_argument("--model", type=str, default="MSTGCNet")
    parser.add_argument('--experiment_mode', choices=['E0','E1','E2','E3','E4','E5'], default='E0')
    parser.add_argument('--d_var', type=int, choices=[16,32], default=16)
    parser.add_argument('--graph_dim', type=int, default=16)
    parser.add_argument('--granger_prior', type=str, default='')
    parser.add_argument('--lambda_prior', type=float, default=1.0)
    parser.add_argument('--prior_shuffle_seed', type=int, default=2025)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--report_diagnostics', type=str2bool, default=False)

    parser.add_argument("--data", type=str, default="ALFA", choices=["ALFA", "FD"])
    parser.add_argument("--root_path", type=str, default="./dataset/ALFA10vars/")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--freq", type=str, default="s")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=0)
    parser.add_argument("--anomaly_ratio", type=float, default=None)
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="atssd",
        choices=["percentile", "atssd", "causal_atssd"],
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        default="causal_last",
        choices=["causal_last", "overlap_mean", "paper_nonoverlap"],
        help=(
            "causal_last scores only the final point of each stride-1 window; "
            "overlap_mean averages every reconstruction covering a point; "
            "paper_nonoverlap follows the released experiment scaffold."
        ),
    )
    parser.add_argument("--paper_strict", type=str2bool, default=True)
    parser.add_argument("--implementation_tag", type=str, default="v12_finegrained")
    parser.add_argument(
        "--score_normalization",
        type=str,
        default="none",
        choices=["none", "train_feature"],
    )

    parser.add_argument("--enc_in", type=int, default=10)
    parser.add_argument("--dec_in", type=int, default=10)
    parser.add_argument("--c_out", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--revin", type=int, default=1)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--des", type=str, default="paper_alfa")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="none")
    parser.add_argument("--use_amp", action="store_true", default=False)

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda", choices=["cuda", "mps"])
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--attn_heads", type=int, default=2)
    parser.add_argument("--num_experts_list", nargs="+", type=int, default=[4, 4, 4])
    parser.add_argument(
        "--patch_size_list",
        nargs="+",
        type=int,
        default=[size for layer in REPRO_PATCH_SIZE_LIST for size in layer],
    )
    parser.add_argument("--noisy_gating", type=int, default=1)
    parser.add_argument("--trend_kernel_sizes", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--seasonality_k", type=int, default=3)
    parser.add_argument("--loss_coef", type=float, default=1e-2)
    parser.add_argument("--winsize", type=int, default=96)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--alarm_confirmation", type=int, default=1)
    parser.add_argument("--latch_alarm", type=int, choices=[0, 1], default=0)
    parser.add_argument("--threshold_adaptation_clip", type=float, default=2.0)
    graph_options = parser.add_mutually_exclusive_group()
    graph_options.add_argument(
        "--abl_GCN", type=int, choices=[0, 1], default=0,
        help="1 skips graph propagation, retaining patch attention and FFN.",
    )
    graph_options.add_argument(
        "--without_graph", dest="abl_GCN", action="store_const", const=1,
        help="CC-STGCN ablation: skip graph propagation (alias for --abl_GCN 1).",
    )
    parser.add_argument("--abl_thre", type=int, default=1)
    parser.add_argument("--residual_connection", type=int, default=1)
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument("--lambda_contrastive", type=float, default=0.0)

    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--p_hidden_dims", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)
    return parser


def validate_paper_args(args):
    expected = {
        "seq_len": 96,
        "winsize": 96,
        "d_model": 64,
        "e_layers": 3,
        "top_k": 3,
        "knn_k": 5,
        "seasonality_k": 3,
        "batch_size": 128,
        "train_epochs": 10,
        "patience": 3,
        "learning_rate": 1e-4,
        "dropout": 0.1,
        "alpha": 0.01,
        "threshold_method": "atssd",
        "score_normalization": "none",
        "alarm_confirmation": 1,
        "latch_alarm": 0,
        "lradj": "none",
    }
    mismatches = []
    for name, paper_value in expected.items():
        actual = getattr(args, name)
        if isinstance(paper_value, float):
            matches = bool(np.isclose(actual, paper_value))
        else:
            matches = actual == paper_value
        if not matches:
            mismatches.append(f"--{name}={actual!r} (paper: {paper_value!r})")

    if args.num_experts_list != [4, 4, 4]:
        mismatches.append(
            f"--num_experts_list={args.num_experts_list!r} (paper: [4, 4, 4])"
        )
    patch_pool = {2, 6, 8, 12, 16, 32}
    if any(len(layer) != 4 for layer in args.patch_size_list):
        mismatches.append("each GMoE block must contain four patch experts")
    if any(size not in patch_pool for layer in args.patch_size_list for size in layer):
        mismatches.append(
            f"--patch_size_list contains values outside paper pool {sorted(patch_pool)}"
        )
    if args.patch_size_list != REPRO_PATCH_SIZE_LIST:
        mismatches.append(
            "--patch_size_list must match the configured coarse-to-fine "
            f"reproduction assignment {REPRO_PATCH_SIZE_LIST!r}"
        )
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            "Paper-strict configuration mismatch:\n  - "
            + details
            + "\nUse --paper_strict false only for ablations or diagnostics."
        )


def normalize_args(args):
    metadata = {}
    metadata_path = os.path.join(args.root_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    args.dataset_split = str(metadata.get("split_policy", "unknown"))
    args.split_seed = metadata.get("split_seed", "na")
    if args.experiment_mode != 'E0':
        if args.abl_GCN:
            raise ValueError('E1--E5 graph mechanisms cannot be combined with --abl_GCN')
        if args.graph_dim < 1 or args.knn_k < 1 or not np.isfinite(args.lambda_prior) or args.lambda_prior < 0:
            raise ValueError('Positive graph_dim/knn_k and finite nonnegative lambda_prior required')
        from scripts.preprocess_alfa import FEATURE_COLUMNS
        args.variable_names = metadata.get('features', FEATURE_COLUMNS)
        if args.data != 'ALFA' or args.enc_in != len(args.variable_names):
            raise ValueError('Variable V1 requires ordered ALFA features')
        if args.experiment_mode in {'E3','E4','E5'}:
            if not args.granger_prior:
                raise ValueError('--granger_prior is required for E3/E4/E5')
            from layers.GrangerPrior import load_granger_prior
            _, prior_meta = load_granger_prior(args.granger_prior, args.variable_names)
            args.granger_prior_sha256 = prior_meta['sha256']
            if not metadata:
                raise ValueError('Formal prior experiments require dataset metadata.json')
            if metadata:
                from pathlib import Path
                actual = hashlib.sha256((Path(args.root_path)/'train.csv').read_bytes()).hexdigest()
                if prior_meta.get('train_sha256') != actual:
                    raise ValueError('Prior training data hash differs from this dataset')
                actual_meta = hashlib.sha256((Path(args.root_path)/'train_meta.csv').read_bytes()).hexdigest()
                if prior_meta.get('train_meta_sha256') != actual_meta:
                    raise ValueError('Prior training metadata hash differs from this dataset')
                if prior_meta.get('sampling_mode') != metadata.get('sampling_mode'):
                    raise ValueError('Prior and model dataset sampling semantics differ')
                if prior_meta.get('sampling_hz') != metadata.get('sampling_hz'):
                    raise ValueError('Prior and model dataset sampling frequencies differ')
    if args.anomaly_ratio is None:
        test_ratio = metadata.get("test_anomaly_ratio")
        args.anomaly_ratio = (
            float(test_ratio) * 100.0 if test_ratio is not None else 25.0
        )

    if torch.cuda.is_available() and args.use_gpu and args.gpu_type == "cuda":
        args.device = torch.device(f"cuda:{args.gpu}")
    elif args.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        args.gpu_type = "mps"
        args.device = torch.device("mps")
    else:
        args.use_gpu = False
        args.device = torch.device("cpu")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device) for device in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    if len(args.num_experts_list) == 1:
        args.num_experts_list = args.num_experts_list * args.e_layers
    if len(args.num_experts_list) != args.e_layers:
        raise ValueError("--num_experts_list length must match --e_layers")

    patch_sizes = np.array(args.patch_size_list)
    if patch_sizes.size % args.e_layers != 0:
        raise ValueError("--patch_size_list length must be divisible by --e_layers")
    args.patch_size_list = patch_sizes.reshape(args.e_layers, -1).tolist()

    if args.threshold_method == "percentile":
        args.abl_thre = 1
    else:
        args.abl_thre = 0

    if args.paper_strict:
        validate_paper_args(args)

    return args


def build_setting(args, iteration):
    excluded = {
        "checkpoints",
        "device",
        "devices",
        "device_ids",
        "gpu",
        "gpu_type",
        "is_training",
        "itr",
        "num_workers",
        "use_gpu",
        "use_multi_gpu",
        "report_diagnostics",
    }
    fingerprint_payload = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }
    if args.experiment_mode == 'E0':
        # Preserve original default fingerprint and checkpoint naming.
        for key in ('experiment_mode','d_var','graph_dim','granger_prior','lambda_prior',
                    'prior_shuffle_seed','report_diagnostics'):
            fingerprint_payload.pop(key, None)
        if args.seed == 2025: fingerprint_payload.pop('seed', None)
    serialized = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    fingerprint = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]
    ablation_suffix = "_no_graph" if args.abl_GCN else ""
    if args.experiment_mode != 'E0': ablation_suffix += '_' + args.experiment_mode
    return (
        f"{args.model_id}_{args.model}_sl{args.seq_len}_dm{args.d_model}"
        f"_el{args.e_layers}_{args.score_mode}_{args.threshold_method}"
        f"_{args.implementation_tag}{ablation_suffix}_cfg{fingerprint}_{iteration}"
    )


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    fix_seed = parsed_args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = normalize_args(parsed_args)

    print("GPU is available:")
    print("cuda", torch.cuda.is_available())
    print("mps", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print("Args in experiment:")
    print_args(args)
    graph_descriptions = {
        'E0': 'original V12 representation and graph',
        'E1': 'variable features + original fixed-embedding graph',
        'E2': 'variable features + input-dependent dynamic graph',
        'E3': 'variable features + fixed historical Granger graph',
        'E4': 'variable features + Granger-guided dynamic graph',
        'E5': 'variable features + shuffled-Granger dynamic graph',
    }
    print('CC-STGCN: graph propagation disabled; patch attention and FFN retained.'
          if args.abl_GCN else f'{args.experiment_mode}: {graph_descriptions[args.experiment_mode]}')

    exp_cls = Exp_Anomaly_Detection
    os.makedirs(args.checkpoints, exist_ok=True)

    for ii in range(args.itr):
        setting = build_setting(args, ii)
        exp = exp_cls(args)
        if args.is_training:
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
        else:
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting, test=1)

        if args.use_gpu and args.gpu_type == "mps":
            torch.backends.mps.empty_cache()
        elif args.use_gpu and args.gpu_type == "cuda":
            torch.cuda.empty_cache()
