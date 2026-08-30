import argparse
import subprocess
import sys


ABLATIONS = {
    "full": [],
    "no_spectral": ["--disable_spectral", "true"],
    "no_dynamic": [
        "--disable_dynamic_score",
        "true",
        "--score_fusion",
        "obs",
        "--dynamic_loss_weight",
        "0.0",
    ],
    "obs_only": ["--score_fusion", "obs"],
    "dyn_only": ["--score_fusion", "dyn"],
    "no_graph": ["--disable_graph", "true"],
    "no_router": ["--disable_router", "true"],
    "no_probabilistic": ["--disable_probabilistic", "true"],
}


def build_base_command(args):
    command = [
        sys.executable,
        "run.py",
        "--model",
        "DTSGAD",
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
        args.score_mode,
        "--threshold_method",
        args.threshold_method,
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
    parser = argparse.ArgumentParser(description="Run DTSGAD ablation experiments.")
    parser.add_argument("--model_id", default="ALFA10vars")
    parser.add_argument("--root_path", default="./dataset/ALFA10vars/")
    parser.add_argument("--use_gpu", default="false")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--score_mode", default="causal_last")
    parser.add_argument("--threshold_method", default="atssd")
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(ABLATIONS.keys()),
        choices=list(ABLATIONS.keys()),
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
            f"dtsgad_{variant}",
        ] + ABLATIONS[variant]
        print("\n===== Running {} =====".format(variant), flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
