import numpy as np
import ctypes
import os

# Load the shared library from the build directory
lib_path = os.path.join(os.path.dirname(__file__), '../cpp/build/libmatrix_metal.dylib')
lib = ctypes.CDLL(lib_path)

# Define the argument and return types
lib.matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.matmul.restype = None

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.dtype == np.float32 and B.dtype == np.float32
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = np.zeros((M, N), dtype=np.float32)
    lib.matmul(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, N, K
    )
    return C
