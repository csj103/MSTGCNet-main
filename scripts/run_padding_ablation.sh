#!/usr/bin/env bash
# Run from a persistent server job/terminal. Both models start from random weights.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs_padding

echo 'Training E0-P (E0 + replicate); see logs_padding/E0-P.log'
python -u run.py --is_training 1 --model MSTGCNet --experiment_mode E0 --input_padding replicate --root_path ./dataset/ALFA10vars_20hz/ --granger_prior ./analysis/variable_granger_v1/prior/granger_prior.npz --d_var 16 --lambda_prior 1 --seed 2025 --score_mode causal_last --threshold_method atssd --use_gpu true --gpu 0 --num_workers 0 --report_diagnostics true > logs_padding/E0-P.log 2>&1

echo 'Training E1-P (E1 + circular); see logs_padding/E1-P.log'
python -u run.py --is_training 1 --model MSTGCNet --experiment_mode E1 --input_padding circular --root_path ./dataset/ALFA10vars_20hz/ --granger_prior ./analysis/variable_granger_v1/prior/granger_prior.npz --d_var 16 --lambda_prior 1 --seed 2025 --score_mode causal_last --threshold_method atssd --use_gpu true --gpu 0 --num_workers 0 --report_diagnostics true > logs_padding/E1-P.log 2>&1

echo 'E0-P and E1-P training/testing finished.'
