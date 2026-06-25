import argparse
import os
import random

import numpy as np
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from utils.print_args import print_args


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

    parser.add_argument("--data", type=str, default="ALFA", choices=["ALFA", "FD"])
    parser.add_argument("--root_path", type=str, default="./dataset/ALFA10vars/")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--freq", type=str, default="s")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=0)
    parser.add_argument("--anomaly_ratio", type=float, default=25.0)
    parser.add_argument(
        "--threshold_method",
        type=str,
        default="atssd",
        choices=["percentile", "atssd"],
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--des", type=str, default="paper_alfa")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type1")
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
        default=[16, 12, 8, 32, 12, 8, 6, 4, 8, 6, 4, 2],
    )
    parser.add_argument("--noisy_gating", type=int, default=1)
    parser.add_argument("--trend_kernel_sizes", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--seasonality_k", type=int, default=3)
    parser.add_argument("--loss_coef", type=float, default=1e-2)
    parser.add_argument("--winsize", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--abl_GCN", type=int, default=0)
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


def normalize_args(args):
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
    elif args.threshold_method == "atssd":
        args.abl_thre = 0

    return args


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
        setting = (
            f"{args.task_name}_{args.model_id}_{args.model}_{args.data}"
            f"_sl{args.seq_len}_dm{args.d_model}_nh{args.n_heads}"
            f"_ke{args.top_k}_kn{args.knn_k}_el{args.e_layers}"
            f"_ws{args.winsize}_ap{args.alpha}_df{args.d_ff}"
            f"_eb{args.embed}_{ii}"
        )
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
