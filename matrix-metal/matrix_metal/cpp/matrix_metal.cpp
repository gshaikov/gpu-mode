// matrix_metal.cpp
// C++ backend for Metal-accelerated matrix multiplication

#include <vector>
#include <stdexcept>
#include <iostream>

#ifdef __APPLE__
extern "C" int matmul_metal(const float* A, const float* B, float* C, int M, int N, int K);
#endif

static void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

extern "C" {
void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
#ifdef __APPLE__
    int status = matmul_metal(A, B, C, M, N, K);
    if (status != 0) {
        std::cerr << "Falling back to CPU matmul due to Metal failure (code " << status << ")" << std::endl;
        cpu_matmul(A, B, C, M, N, K);
    }
#else
    cpu_matmul(A, B, C, M, N, K);
#endif
}
}
