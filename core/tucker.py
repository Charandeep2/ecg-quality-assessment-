"""
Higher-Order Tensor Decomposition (Tucker) and Adaptive Rank Estimation.
"""

import ast
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import tensorly as tl
    from tensorly.decomposition import tucker
    tl.set_backend("pytorch")
except ImportError:
    tl = None
    tucker = None

from core.config import ENERGY_THRESHOLD, FIXED_RANK, MAX_RANK

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    """Return the active computing device (CUDA or CPU)."""
    return _DEVICE


def _check_tensorly():
    if tl is None or tucker is None:
        raise ImportError(
            "tensorly is required for Tucker decomposition. "
            "Install it using: pip install tensorly"
        )


def estimate_energy_rank(
    tensor: np.ndarray,
    max_rank: Tuple[int, int, int] = MAX_RANK,
    energy_threshold: float = ENERGY_THRESHOLD,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, int]:
    """
    Estimate adaptive Tucker rank for each tensor mode based on singular-value cumulative energy.
    Truncates at the index where cumulative energy meets or exceeds energy_threshold (0.95),
    bounded by max_rank.
    """
    dev = device or _DEVICE
    tensor_t = torch.as_tensor(tensor, dtype=torch.float32, device=dev)
    ranks = []

    with torch.no_grad():
        for mode, rank_cap in enumerate(max_rank):
            unfolded = (
                torch.movedim(tensor_t, mode, 0)
                .reshape(tensor_t.shape[mode], -1)
                .contiguous()
            )
            component_count = min(rank_cap, min(unfolded.shape))

            if component_count == min(unfolded.shape):
                singular_values = torch.linalg.svdvals(unfolded)
            else:
                _, singular_values, _ = torch.pca_lowrank(
                    unfolded,
                    q=component_count,
                    center=False,
                    niter=2,
                )
                singular_values = torch.sort(
                    singular_values, descending=True
                ).values

            total_energy = torch.sum(unfolded.square())
            energy = singular_values.square()

            if total_energy.item() <= 0:
                rank = 1
            else:
                cumulative = torch.cumsum(energy, dim=0) / (total_energy + 1e-12)
                reached = torch.nonzero(cumulative >= energy_threshold, as_tuple=False)
                rank = (
                    int(reached[0].item()) + 1
                    if reached.numel()
                    else component_count
                )

            ranks.append(max(1, min(rank, rank_cap)))
            del unfolded, singular_values, energy, total_energy

    del tensor_t
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return tuple(ranks)


def calculate_tucker(
    tensor: np.ndarray,
    rank: Tuple[int, int, int] = FIXED_RANK,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute Tucker tensor decomposition:
    X ≈ G ×_1 U_1 ×_2 U_2 ×_3 U_3

    Args:
        tensor (np.ndarray): 3D input tensor.
        rank (tuple): Tucker core rank (R1, R2, R3). Default: (50, 20, 6).
        device (torch.device, optional): PyTorch compute device.

    Returns:
        core (np.ndarray): Tucker core tensor of shape (R1, R2, R3).
        factors (list of np.ndarray): Factor matrices [U1, U2, U3].
    """
    _check_tensorly()
    dev = device or _DEVICE
    tensor_t = torch.as_tensor(tensor, dtype=torch.float32, device=dev)

    with torch.no_grad():
        core_t, factors_t = tucker(
            tensor_t,
            rank=list(rank),
            init="svd",
            tol=1e-5,
        )

    # Detach to NumPy arrays
    core = core_t.detach().cpu().numpy()
    factors = [factor.detach().cpu().numpy() for factor in factors_t]

    del tensor_t, core_t, factors_t
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return core, factors


def calculate_reconstruction_error(
    tensor: np.ndarray,
    core: np.ndarray,
    factors: List[np.ndarray],
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute relative Frobenius norm reconstruction error between the original
    tensor and its Tucker reconstruction:
    Error = ||X - X_hat||_F / ||X||_F
    """
    _check_tensorly()
    dev = device or _DEVICE
    tensor_t = torch.as_tensor(tensor, dtype=torch.float32, device=dev)
    core_t = torch.as_tensor(core, dtype=torch.float32, device=dev)
    factors_t = [
        torch.as_tensor(f, dtype=torch.float32, device=dev) for f in factors
    ]

    with torch.no_grad():
        reconstructed_t = tl.tucker_to_tensor((core_t, factors_t))
        error = torch.linalg.vector_norm(tensor_t - reconstructed_t) / (
            torch.linalg.vector_norm(tensor_t) + 1e-12
        )

    result = float(error.item())
    del tensor_t, core_t, factors_t, reconstructed_t

    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return result


def representation_size(shape: Tuple[int, ...], rank: Tuple[int, ...]) -> int:
    """
    Calculate the total number of stored parameters in a Tucker decomposition:
    Size = prod(rank) + sum(shape[i] * rank[i])
    """
    core_params = int(np.prod(rank))
    factor_params = sum(shape[i] * rank[i] for i in range(len(shape)))
    return core_params + factor_params


def compression_ratio(shape: Tuple[int, ...], rank: Tuple[int, ...]) -> float:
    """
    Calculate data compression ratio:
    Ratio = (original tensor elements) / (Tucker parameters)
    """
    original_size = float(np.prod(shape))
    stored_size = float(representation_size(shape, rank))
    return original_size / max(stored_size, 1.0)


def parse_saved_rank(
    value: Union[str, tuple, list, np.ndarray],
    max_rank: Tuple[int, int, int] = MAX_RANK,
) -> Tuple[int, int, int]:
    """
    Parse a string, list, or tuple rank into a validated 3-tuple integer rank.
    """
    if isinstance(value, (tuple, list, np.ndarray)):
        rank = tuple(int(item) for item in value)
    else:
        rank = tuple(int(item) for item in ast.literal_eval(str(value)))

    if len(rank) != 3:
        raise ValueError("Saved Tucker rank must have exactly 3 dimensions.")

    if any(item < 1 or item > cap for item, cap in zip(rank, max_rank)):
        raise ValueError(f"Rank elements {rank} outside allowed bounds (1 to {max_rank}).")

    return rank

