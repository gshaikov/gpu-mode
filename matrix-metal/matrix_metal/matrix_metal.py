import numpy as np
from matrix_metal import _matrix_metal

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication using Metal (or CPU fallback).
    Args:
        A: np.ndarray of shape (M, K), dtype float32
        B: np.ndarray of shape (K, N), dtype float32
    Returns:
        np.ndarray of shape (M, N), dtype float32
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    return _matrix_metal.matmul(A, B)
