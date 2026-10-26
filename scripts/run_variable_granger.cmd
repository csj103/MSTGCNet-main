@echo off
setlocal
pushd "%~dp0.."
set "MODE=%~1"
if "%MODE%"=="" set "MODE=E4"
if /I "%MODE%"=="--help" (
  echo Usage: scripts\run_variable_granger.cmd E0^|E1^|E2^|E3^|E4^|E5^|ALL true^|false
  popd
  exit /b 0
)
set "USE_GPU=%~2"
if "%USE_GPU%"=="" set "USE_GPU=true"
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
if not exist "dataset\ALFA10vars\metadata.json" (
  python scripts\preprocess_alfa.py --source alfa_10vars --output dataset\ALFA10vars --split-policy fine_grained --seed 2025
  if errorlevel 1 goto failed
)
if not exist "dataset\ALFA10vars_20hz\metadata.json" (
  python scripts\regularize_alfa.py --source dataset\ALFA10vars --output dataset\ALFA10vars_20hz --hz 20
  if errorlevel 1 goto failed
)
if not exist "analysis\variable_granger_v1\prior\granger_prior.npz" (
  python scripts\build_granger_prior.py --root dataset\ALFA10vars_20hz --max-lag 5
  if errorlevel 1 goto failed
)
if /I "%MODE%"=="ALL" (
  for %%M in (E0 E1 E2 E3 E4 E5) do (
    python run.py --model MSTGCNet --experiment_mode %%M --root_path .\dataset\ALFA10vars_20hz\ --granger_prior .\analysis\variable_granger_v1\prior\granger_prior.npz --d_var 16 --lambda_prior 1 --score_mode causal_last --threshold_method atssd --seed 2025 --report_diagnostics true --use_gpu %USE_GPU% --num_workers 0
    if errorlevel 1 goto failed
  )
) else (
  python run.py --model MSTGCNet --experiment_mode %MODE% --root_path .\dataset\ALFA10vars_20hz\ --granger_prior .\analysis\variable_granger_v1\prior\granger_prior.npz --d_var 16 --lambda_prior 1 --score_mode causal_last --threshold_method atssd --seed 2025 --report_diagnostics true --use_gpu %USE_GPU% --num_workers 0
  if errorlevel 1 goto failed
)
popd
exit /b 0
:failed
popd
exit /b 1
