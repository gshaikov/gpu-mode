// matrix_metal.cpp
// C++ backend for Metal-accelerated matrix multiplication

#include <vector>
#include <stdexcept>

extern "C" {
// Placeholder for Metal-accelerated matrix multiplication
void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: Implement Metal-based matrix multiplication
    // For now, use CPU fallback
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
}
