"""Ordered, directed training-only VAR prior and patch-major expansion."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def load_granger_prior(path, features, allow_synthetic=False):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        matrix = np.asarray(archive['matrix'], dtype=np.float32)
        metadata = json.loads(str(archive['metadata'].item()))
    if metadata.get('features') != list(features):
        raise ValueError('Granger variable order does not match dataset feature order')
    if metadata.get('synthetic_test') and not allow_synthetic:
        raise ValueError('Synthetic test priors are forbidden in formal experiments')
    if metadata.get('orientation') != 'target_source':
        raise ValueError('Granger orientation must be target_source')
    if metadata.get('method') != 'linear_var' or metadata.get('training_only') is not True:
        raise ValueError('A training-only linear_var prior is required')
    if matrix.shape != (len(features), len(features)):
        raise ValueError('Invalid Granger matrix shape')
    if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError('Granger weights must be finite in [0,1]')
    if int(metadata.get('max_lag', 0)) < 1:
        raise ValueError('Granger max_lag must be positive')
    if np.any(np.diag(matrix) != 0):
        raise ValueError('Prior diagonal must be zero; self-loops are added separately')
    metadata['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return torch.from_numpy(matrix.copy()), metadata


def expand_prior(matrix, num_patches, patch_size, max_lag):
    """G[target,source], historical patches only, intersecting sample-lag support.

    A delta-patch pair has possible time distances [(delta-1)*p+1,(delta+1)*p-1].
    This coarse support test does not claim a precise causal delay for a patch.
    Within-patch associations never receive a lagged Granger bias.
    """
    num_vars = matrix.shape[0]
    patch = torch.arange(num_patches, device=matrix.device).repeat_interleave(num_vars)
    variable = torch.arange(num_vars, device=matrix.device).repeat(num_patches)
    delta = patch[:, None] - patch[None, :]
    historical = (delta > 0) & ((delta - 1) * patch_size + 1 <= max_lag)
    return matrix[variable[:, None], variable[None, :]] * historical
