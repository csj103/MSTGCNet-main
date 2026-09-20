import argparse
import subprocess
import sys


VARIANTS = {
    "mean_atssd": [
        "--score_aggregation",
        "mean",
        "--threshold_method",
        "atssd",
    ],
    "topk_atssd": [
        "--score_aggregation",
        "topk",
        "--score_topk",
        "3",
        "--threshold_method",
        "atssd",
    ],
    "train_feature_topk_atssd": [
        "--score_aggregation",
        "train_feature_topk",
        "--score_topk",
        "3",
        "--threshold_method",
        "atssd",
    ],
    "train_feature_topk_dual": [
        "--score_aggregation",
        "train_feature_topk",
        "--score_topk",
        "3",
        "--threshold_method",
        "dual_time",
        "--dual_short_quantile",
        "0.995",
        "--dual_long_quantile",
        "0.990",
        "--dual_ewma_alpha",
        "0.050",
    ],
    "max_dual": [
        "--score_aggregation",
        "max",
        "--threshold_method",
        "dual_time",
        "--dual_short_quantile",
        "0.995",
        "--dual_long_quantile",
        "0.990",
        "--dual_ewma_alpha",
        "0.050",
    ],
}


def build_base_command(args):
    command = [
        sys.executable,
        "run.py",
        "--model",
        "MSTGCNet",
        "--model_id",
        args.model_id,
        "--root_path",
        args.root_path,
        "--use_gpu",
        str(args.use_gpu).lower(),
        "--num_workers",
        str(args.num_workers),
        "--paper_strict",
        "false",
        "--seq_len",
        str(args.seq_len),
        "--winsize",
        str(args.seq_len),
        "--score_mode",
        "causal_last",
        "--train_epochs",
        str(args.train_epochs),
        "--batch_size",
        str(args.batch_size),
        "--d_model",
        str(args.d_model),
        "--d_ff",
        str(args.d_ff),
        "--e_layers",
        str(args.e_layers),
        "--learning_rate",
        str(args.learning_rate),
    ]
    extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
    if extra:
        command.extend(extra)
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Run MSTGCNet score and threshold ablation experiments."
    )
    parser.add_argument("--model_id", default="ALFA10vars")
    parser.add_argument("--root_path", default="./dataset/ALFA10vars/")
    parser.add_argument("--use_gpu", default="false")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are forwarded to run.py.",
    )
    args = parser.parse_args()

    base = build_base_command(args)
    for variant in args.variants:
        command = base + [
            "--implementation_tag",
            f"mstgcnet_score_{variant}",
        ] + VARIANTS[variant]
        print("\n===== Running {} =====".format(variant), flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
