// matrix_metal.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void matmul(const float* A, const float* B, float* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif
