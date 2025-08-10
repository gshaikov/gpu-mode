import numpy as np
from matrix_metal.matrix_metal import matmul

A = np.random.rand(4, 8).astype(np.float32)
B = np.random.rand(8, 5).astype(np.float32)
C = matmul(A, B)
print("A:", A)
print("B:", B)
print("C:", C)
