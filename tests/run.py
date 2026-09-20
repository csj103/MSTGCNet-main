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
    parser.add_argument(
        "--model",
        type=str,
        default="MSTGCNet",
        choices=["MSTGCNet", "DTSGAD"],
    )

    parser.add_argument("--data", type=str, default="ALFA", choices=["ALFA", "FD"])
    parser.add_argument("--root_path", type=str, default="./dataset/ALFA10vars/")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--freq", type=str, default="s")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument(
        "--checkpoint_setting",
        type=str,
        default="",
        help=(
            "Optional existing checkpoint folder to load when --is_training 0. "
            "Useful for post-hoc scoring diagnostics such as overlap_mean."
        ),
    )

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=0)
    parser.add_argument("--anomaly_ratio", type=float, default=None)
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="atssd",
        choices=[
            "percentile",
            "atssd",
            "causal_atssd",
            "oracle",
            "val_quantile",
            "lagged_atssd",
            "freeze_atssd",
            "median_mad",
        ],
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
    parser.add_argument(
        "--save_train_feature_scale",
        type=str2bool,
        default=False,
        help=(
            "Save per-feature normal training errors for post-hoc score "
            "diagnostics without changing the scoring mode."
        ),
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
    parser.add_argument("--val_quantile", type=float, default=0.995)
    parser.add_argument(
        "--val_quantiles",
        nargs="+",
        type=float,
        default=[0.95, 0.975, 0.99, 0.995, 0.999],
    )
    parser.add_argument("--mad_k", type=float, default=5.0)
    parser.add_argument(
        "--mad_ks",
        nargs="+",
        type=float,
        default=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    )
    parser.add_argument("--threshold_diagnostics", type=str2bool, default=True)
    parser.add_argument("--abl_GCN", type=int, default=0)
    parser.add_argument("--abl_thre", type=int, default=1)
    parser.add_argument("--residual_connection", type=int, default=1)
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument("--lambda_contrastive", type=float, default=0.0)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument(
        "--dtsgad_objective",
        type=str,
        default="reconstruction",
        choices=["reconstruction", "next_step_prediction"],
        help="DTSGAD training objective.",
    )
    parser.add_argument(
        "--dtsgad_loss_reduction",
        type=str,
        default="sum",
        choices=["sum", "mean"],
        help=(
            "DTSGAD reconstruction loss reduction. Use sum for legacy runs; "
            "use mean for A4 loss alignment."
        ),
    )
    parser.add_argument(
        "--target_horizon",
        type=int,
        default=1,
        help="Prediction horizon for --dtsgad_objective next_step_prediction.",
    )
    parser.add_argument("--obs_topk", type=int, default=3)
    parser.add_argument(
        "--score_fusion",
        type=str,
        default="dual",
        choices=["dual", "obs", "dyn"],
        help="DTSGAD score fusion: observation only, dynamic only, or both.",
    )
    parser.add_argument("--dynamic_score_weight", type=float, default=0.05)
    parser.add_argument("--dynamic_loss_weight", type=float, default=0.01)
    parser.add_argument("--last_loss_weight", type=float, default=0.2)
    parser.add_argument("--balance_loss_weight", type=float, default=1e-2)
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.0,
        help="Random channel mask ratio for DTSGAD masked reconstruction training.",
    )
    parser.add_argument(
        "--mask_eval_mode",
        type=str,
        default="none",
        choices=["none", "channelwise"],
        help="DTSGAD masked reconstruction scoring mode.",
    )
    parser.add_argument("--spectral_temperature", type=float, default=0.2)
    parser.add_argument("--disable_spectral", type=str2bool, default=False)
    parser.add_argument("--disable_dynamic_score", type=str2bool, default=False)
    parser.add_argument(
        "--graph_mode",
        type=str,
        default="learned",
        choices=["learned", "identity", "off"],
        help="DTSGAD graph ablation mode.",
    )
    parser.add_argument(
        "--graph_residual_mode",
        type=str,
        default="shared",
        choices=["shared", "expert", "mixed"],
        help=(
            "DTSGAD graph residual source: shared block input, expert pre-graph "
            "state, or alpha-mixed shared/expert state."
        ),
    )
    parser.add_argument(
        "--graph_residual_alpha",
        type=float,
        default=1.0,
        help="Shared-state weight for --graph_residual_mode mixed.",
    )
    parser.add_argument("--disable_graph", type=str2bool, default=False)
    parser.add_argument("--disable_router", type=str2bool, default=False)
    parser.add_argument("--disable_probabilistic", type=str2bool, default=False)
    parser.add_argument("--router_diagnostics", type=str2bool, default=False)
    parser.add_argument("--single_expert_diagnostics", type=str2bool, default=True)
    parser.add_argument(
        "--latent_diagnostics",
        type=str2bool,
        default=False,
        help=(
            "Run post-hoc normal-center latent diagnostics during testing. "
            "This does not affect training or thresholding."
        ),
    )

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

    if args.model != "MSTGCNet":
        args.paper_strict = False

    if bool(getattr(args, "disable_graph", False)) and args.graph_mode == "learned":
        args.graph_mode = "identity"

    if not 0.0 <= float(args.graph_residual_alpha) <= 1.0:
        raise ValueError("--graph_residual_alpha must be in [0, 1]")
    if not 0.0 <= float(args.mask_ratio) < 1.0:
        raise ValueError("--mask_ratio must be in [0, 1)")
    if args.mask_ratio <= 0.0:
        args.mask_eval_mode = "none"
    if int(args.target_horizon) < 1:
        raise ValueError("--target_horizon must be >= 1")
    if args.dtsgad_objective == "next_step_prediction":
        if args.score_mode != "causal_last":
            raise ValueError(
                "next_step_prediction currently supports only causal_last scoring"
            )
        if args.mask_ratio > 0.0:
            raise ValueError("M2 next_step_prediction must not be mixed with M1 masking")
        args.disable_probabilistic = True
        args.disable_dynamic_score = True
        args.score_fusion = "obs"
        args.dynamic_loss_weight = 0.0
        args.last_loss_weight = 0.0
        args.balance_loss_weight = 0.0

    if args.paper_strict:
        validate_paper_args(args)

    return args


def build_setting(args, iteration):
    excluded = {
        "checkpoints",
        "checkpoint_setting",
        "device",
        "devices",
        "device_ids",
        "gpu",
        "gpu_type",
        "is_training",
        "itr",
        "latent_diagnostics",
        "num_workers",
        "mad_ks",
        "router_diagnostics",
        "single_expert_diagnostics",
        "save_train_feature_scale",
        "threshold_diagnostics",
        "use_gpu",
        "use_multi_gpu",
        "val_quantiles",
    }
    fingerprint_payload = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded
    }
    if fingerprint_payload.get("graph_mode") == "learned":
        fingerprint_payload.pop("graph_mode")
    if fingerprint_payload.get("graph_residual_mode") == "shared":
        fingerprint_payload.pop("graph_residual_mode")
    if fingerprint_payload.get("graph_residual_mode") != "mixed":
        fingerprint_payload.pop("graph_residual_alpha", None)
    if float(fingerprint_payload.get("mask_ratio", 0.0)) <= 0.0:
        fingerprint_payload.pop("mask_ratio", None)
        fingerprint_payload.pop("mask_eval_mode", None)
    if fingerprint_payload.get("dtsgad_objective") == "reconstruction":
        fingerprint_payload.pop("dtsgad_objective", None)
        fingerprint_payload.pop("target_horizon", None)
    if fingerprint_payload.get("dtsgad_loss_reduction") == "sum":
        fingerprint_payload.pop("dtsgad_loss_reduction", None)
    serialized = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    fingerprint = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]
    return (
        f"{args.model_id}_{args.model}_sl{args.seq_len}_dm{args.d_model}"
        f"_el{args.e_layers}_{args.score_mode}_{args.threshold_method}"
        f"_{args.implementation_tag}_cfg{fingerprint}_{iteration}"
    )


if __name__ == "__main__":
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = normalize_args(build_parser().parse_args())

    print("GPU is available:")
    print("cuda", torch.cuda.is_available())
    print("mps", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print("Args in experiment:")
    print_args(args)

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
