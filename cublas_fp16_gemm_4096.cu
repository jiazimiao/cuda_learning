#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err));              \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t st = (call);                                        \
        if (st != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",        \
                         __FILE__, __LINE__, static_cast<int>(st));        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

static float rand_float()
{
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
}

int main(int argc, char **argv)
{
    const int N = (argc > 1) ? std::atoi(argv[1]) : 4096;
    const int warmup_iters = 10;
    const int timing_iters = 100;

    const size_t elems = static_cast<size_t>(N) * N;
    const size_t half_bytes = elems * sizeof(half);
    const size_t float_bytes = elems * sizeof(float);

    std::vector<half> hA(elems);
    std::vector<half> hB(elems);
    std::vector<float> hC(elems);

    std::srand(0);
    for (size_t i = 0; i < elems; ++i) {
        hA[i] = __float2half(rand_float());
        hB[i] = __float2half(rand_float());
    }

    half *dA = nullptr;
    half *dB = nullptr;
    float *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, half_bytes));
    CUDA_CHECK(cudaMalloc(&dB, half_bytes));
    CUDA_CHECK(cudaMalloc(&dC, float_bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, float_bytes));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major matrices. This computes C_col = A_col * B_col.
    // For row-major buffers, it is equivalent to C_row = B_row * A_row.
    // Timing is identical for square GEMM, which is what this benchmark needs.
    for (int i = 0; i < warmup_iters; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            dA, CUDA_R_16F, N,
            dB, CUDA_R_16F, N,
            &beta,
            dC, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timing_iters; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            dA, CUDA_R_16F, N,
            dB, CUDA_R_16F, N,
            &beta,
            dC, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double avg_ms = static_cast<double>(total_ms) / timing_iters;
    const double flops = 2.0 * N * N * static_cast<double>(N);
    const double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

    std::printf("N=%d FP16 input, FP32 accumulate, FP32 output\n", N);
    std::printf("Average cuBLAS GEMM time: %.4f ms\n", avg_ms);
    std::printf("Throughput: %.2f TFLOP/s\n", tflops);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, float_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
