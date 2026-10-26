# H20 MIG 35 GB / 4 CPU server environment

Scope: MSTGCNet exp/variable-granger-v1, E0--E5, preprocessing, linear VAR prior,
tests and report generation. Commands below assume Linux x86_64 with Conda.
The reported allocation is one NVIDIA-H20-3e MIG instance (2g.35gb), four CPUs.
Use one visible CUDA device (`--gpu 0`); no multi-GPU option is needed.

## Dependencies

| Package | Version | Purpose / verification status |
|---|---|---|
| Python | 3.12 | Local CPU validation used 3.12.7 |
| torch | 2.12.1, CUDA 12.6 wheel candidate | Local tests used 2.12.1+cpu; server CUDA build untested |
| numpy | 1.26.4 | Data, scores; locally imported and tested |
| pandas | 2.2.2 | ALFA CSV and metadata; locally imported and tested |
| scipy | 1.13.1 | Granger F distribution; locally imported and tested |
| scikit-learn | 1.5.1 | Scaling, ROC-AUC/AP/P/R/F1; locally imported and tested |
| matplotlib | 3.9.2 | Prior/dynamic graphs and score plots; locally imported and tested |
| pytest | 7.4.4 | Regression suite; locally tested |
| einops | 0.8.0 | Legacy utils/Other.py only, unused by E0--E5 and not locally tested |

PyTorch's required Python/CUDA runtime dependencies, and the numerical libraries'
dependencies, are installed by pip. This is a direct-dependency pin list, not a
complete transitive/environment lock. Save pip freeze after installation.
There are no custom CUDA extensions in this project. E0--E5 do not require
torchvision, torchaudio, TensorFlow, DGL, torch-geometric, statsmodels, cuML,
Jupyter or a display server. Linux Git is needed by provenance and baseline checks.
Python modules such as argparse/json/hashlib/pathlib/random are standard library.

## 1. Check the host before installing GPU PyTorch

```bash
nvidia-smi
uname -m
ldd --version | head -n 1
```

The default below is the official torch 2.12.1 CUDA 12.6 build, whose Linux
x86_64 wheels use manylinux_2_28. Require glibc >=2.28; Ubuntu 22.04/24.04 are
suitable examples. Do not try upgrading glibc inside a shared host to resolve
an incompatible image; use an appropriate server image instead.

The H20 allocation alone does not reveal the installed NVIDIA driver version.
As a conservative native-CUDA-12.6 setup, use a driver >=560.28.03. NVIDIA also
documents CUDA 12.x minor compatibility with older drivers, with feature/PTX
limitations; >=560.28.03 is not asserted to be the universal minimum for every
CUDA-12 application. If the driver's version is older, determine the wheel/image
from the actual nvidia-smi output rather than reinstalling the shared driver.
The CUDA Version field in nvidia-smi is a driver compatibility indication, not
the Python environment's installed PyTorch CUDA runtime.

Official references checked for these commands:

- [PyTorch 2.12.1 installation commands](https://pytorch.org/get-started/previous-versions/)
- [CUDA 12.6 PyTorch wheels](https://download.pytorch.org/whl/cu126/torch/)
- [CUDA 12.6 driver versions](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html)
- [CUDA minor compatibility and its limits](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

## 2. Create a clean environment

If the platform already supplies Conda, no root/admin install is needed:

```bash
conda create -n mstgcnet-granger python=3.12 pip -y
conda activate mstgcnet-granger
python -m pip install --upgrade pip setuptools wheel
```

If Conda is unavailable but the image has Python 3.12 and its venv module:

```bash
python3.12 -m venv ~/venvs/mstgcnet-granger
source ~/venvs/mstgcnet-granger/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Select one environment mechanism. All subsequent pip commands use that
environment's `python -m pip`. Do not copy the local Anaconda site-packages.

## 3. Clone code, install direct dependencies, then GPU PyTorch

```bash
git clone --branch exp/variable-granger-v1 --single-branch https://github.com/csj103/MSTGCNet-main.git
cd MSTGCNet-main
python -m pip install -r requirements-server.txt
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip check
```

The requirements file/document must first be pushed from the local experimental
branch or copied to the server. An equivalent install independent of that file:

```bash
python -m pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.5.1 matplotlib==3.9.2 pytest==7.4.4 einops==0.8.0
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
```

Use the official cu126 index for the torch command. A separate CUDA Toolkit,
nvcc or manual cuDNN installation is not needed for this project's prebuilt
wheel workflow; the host must expose its NVIDIA driver/GPU to the environment.
Do not mix cu126/cu130/CPU torch builds in this environment.

## 4. Verify imports, NumPy tensor interop and an actual CUDA kernel

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg
python - <<'PY'
import sys
import numpy as np
import pandas as pd
import scipy, sklearn, matplotlib, pytest, einops, torch
print('Python:', sys.version)
for module in [np, pd, scipy, sklearn, matplotlib, pytest, einops, torch]:
    print(module.__name__, module.__version__)
assert torch.cuda.is_available(), 'CUDA unavailable: check driver/container visibility/torch build'
print('PyTorch CUDA:', torch.version.cuda)
print('Visible CUDA devices:', torch.cuda.device_count())
print('Device 0:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
print('Wheel architectures:', torch.cuda.get_arch_list())
x = torch.from_numpy(np.ones((32,32), dtype=np.float32)).to('cuda:0')
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('NumPy -> tensor -> CUDA matmul: OK')
PY
```

Do not overwrite a platform-supplied MIG UUID in CUDA_VISIBLE_DEVICES. Use the
visible device numbered 0 by PyTorch. MIG visibility is managed by the platform.
The existing --num_workers 0 works with the four-CPU allocation.

## 5. Recreate the same data grid and a matching prior on a fresh server clone

Run these in a fresh project with no existing output dataset directories:

```bash
python scripts/preprocess_alfa.py --source alfa_10vars --output dataset/ALFA10vars --split-policy fine_grained --seed 2025
python scripts/regularize_alfa.py --source dataset/ALFA10vars --output dataset/ALFA10vars_20hz --hz 20
python scripts/build_granger_prior.py --root dataset/ALFA10vars_20hz --output analysis/variable_granger_v1/prior --max-lag 5
python -m pytest tests -q --import-mode=importlib
```

If reusing a previously prepared dataset directory, skip preprocessing and
regularization. The regularizer deliberately refuses to overwrite an existing
output directory. Rebuilding the prior is required if its source dataset hashes
do not match: Windows/Linux CSV byte serialization or package differences can
change hashes despite matching row counts. Do not bypass this check. Preserve
and use the same dataset and newly computed prior across all E0--E5 comparisons.

The repository's source ALFA CSVs are tracked, but generated dataset/ is ignored.
Copying a prepared dataset plus its exact prior is also supported. A shallow
clone may omit baseline commit 5382d528 and break the optional baseline checker;
the clone command above intentionally uses full history.

## 6. Actual model forward/backward on the allocated GPU

After preparing data:

```bash
python - <<'PY'
import torch
from run import build_parser, normalize_args
from models.MSTGCNet import Model
args = normalize_args(build_parser().parse_args([
    '--experiment_mode','E4', '--root_path','./dataset/ALFA10vars_20hz/',
    '--granger_prior','./analysis/variable_granger_v1/prior/granger_prior.npz',
    '--use_gpu','true', '--gpu','0'
]))
assert args.use_gpu and args.device.type == 'cuda'
model = Model(args).to(args.device).train()
x = torch.randn(2,96,10,device=args.device)
y, aux = model(x)
loss = (y-x).square().flatten(1).sum(1).mean() + aux
loss.backward()
assert y.shape == x.shape and torch.isfinite(loss)
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
torch.cuda.synchronize()
print('E4 CUDA forward/backward OK:', tuple(y.shape), float(loss.detach()))
PY
```

These are server verification commands for the user to execute. They have not
been run on the user's H20 from the local CPU-only host.

## 7. Train on Linux

The Windows .cmd launcher is not a Linux shell script. Use Python directly:

```bash
mkdir -p logs
python -m pip freeze > logs/environment-server.lock.txt
nvidia-smi > logs/nvidia-smi.txt
git rev-parse HEAD > logs/git-commit.txt
python -u run.py --model MSTGCNet --experiment_mode E4 --root_path ./dataset/ALFA10vars_20hz/ --granger_prior ./analysis/variable_granger_v1/prior/granger_prior.npz --d_var 16 --lambda_prior 1 --seed 2025 --score_mode causal_last --threshold_method atssd --use_gpu true --gpu 0 --num_workers 0 --report_diagnostics true > logs/E4.log 2>&1
```

This is foreground training, with output saved to logs/E4.log. Run it from a
platform-managed persistent terminal/job if sessions can disconnect. Training
and testing run sequentially using the selected best validation checkpoint.
Default batch=128 and at most 10 epochs are retained; GPU batch=128 has not
been measured. If memory is insufficient, use --batch_size 32 --paper_strict
false and apply that same change to every comparison.

For E0--E5, change --experiment_mode only while keeping the shared data/prior and
training settings. Metric files are saved under label_results/<setting>/causal_last/,
weights/config/routing statistics under checkpoints/<setting>/.

## Verification boundary

Local CPU environment: Python 3.12.7, torch 2.12.1+cpu and the pinned core packages
were queried directly; the core model/Granger/test pipeline was previously
verified. einops is an additional legacy-helper dependency, not installed or
required in those CPU tests. No NVIDIA driver was changed and no CUDA package
was installed into the shared local environment. Server CUDA compatibility,
MIG visibility, batch memory and full training still require the checks above.
