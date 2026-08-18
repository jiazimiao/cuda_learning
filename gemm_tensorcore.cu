#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>

using namespace nvcuda::wmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#define CUDA_CHECK(call)                                                                 \
    do                                                                                   \
    {                                                                                    \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess)                                                          \
        {                                                                                \
            std::printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                \
                        cudaGetErrorString(err));                                        \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

static inline int round_up(int value, int multiple)
{
    return ((value + multiple - 1) / multiple) * multiple;
}

static float rand_float()
{
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
}

__global__ void gemm_tensorcore(
    const half* A,
    const half* B,
    float* C,
    int N)
{
    const int tile_m = blockIdx.y * WMMA_M;
    const int tile_n = blockIdx.x * WMMA_N;

    __shared__ half sA[WMMA_M][WMMA_K];
    __shared__ half sB[WMMA_K][WMMA_N];
    __shared__ float sC[WMMA_M][WMMA_N];

    const int lane = threadIdx.x;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for (int tile_k = 0; tile_k < N; tile_k += WMMA_K)
    {
        for (int idx = lane; idx < WMMA_M * WMMA_K; idx += warpSize)
        {
            const int r = idx / WMMA_K;
            const int c = idx % WMMA_K;
            const int g_row = tile_m + r;
            const int g_col = tile_k + c;
            sA[r][c] = (g_row < N && g_col < N) ? A[g_row * N + g_col] : __float2half(0.0f);
        }

        for (int idx = lane; idx < WMMA_K * WMMA_N; idx += warpSize)
        {
            const int r = idx / WMMA_N;
            const int c = idx % WMMA_N;
            const int g_row = tile_k + r;
            const int g_col = tile_n + c;
            sB[r][c] = (g_row < N && g_col < N) ? B[g_row * N + g_col] : __float2half(0.0f);
        }

        __syncwarp();

        load_matrix_sync(a_frag, &sA[0][0], WMMA_K);
        load_matrix_sync(b_frag, &sB[0][0], WMMA_N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncwarp();
    }

    store_matrix_sync(&sC[0][0], c_frag, WMMA_N, mem_row_major);

    __syncwarp();

    for (int idx = lane; idx < WMMA_M * WMMA_N; idx += warpSize)
    {
        const int r = idx / WMMA_N;
        const int c = idx % WMMA_N;
        const int g_row = tile_m + r;
        const int g_col = tile_n + c;
        if (g_row < N && g_col < N)
        {
            C[g_row * N + g_col] = sC[r][c];
        }
    }
}

int main(int argc, char** argv)
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (N <= 0)
    {
        std::printf("Invalid matrix size: %d\n", N);
        return 1;
    }

    const size_t elems = static_cast<size_t>(N) * N;
    const size_t half_bytes = elems * sizeof(half);
    const size_t float_bytes = elems * sizeof(float);

    std::vector<float> hA(N * N);
    std::vector<float> hB(N * N);
    std::vector<float> hC_ref(N * N, 0.0f);
    std::vector<float> hC(N * N, 0.0f);
    std::vector<half> hA_half(elems);
    std::vector<half> hB_half(elems);

    for (int i = 0; i < N * N; ++i)
    {
        hA[i] = rand_float();
        hB[i] = rand_float();
    }

    for (int i = 0; i < N * N; ++i)
    {
        hA_half[i] = __float2half(hA[i]);
        hB_half[i] = __float2half(hB[i]);
    }

    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
            {
                sum += hA[row * N + k] * hB[k * N + col];
            }
            hC_ref[row * N + col] = sum;
        }
    }

    half* dA = nullptr;
    half* dB = nullptr;
    float* dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, half_bytes));
    CUDA_CHECK(cudaMalloc(&dB, half_bytes));
    CUDA_CHECK(cudaMalloc(&dC, float_bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA_half.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB_half.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, float_bytes));

    dim3 block(32, 1, 1);
    dim3 grid((N + WMMA_N - 1) / WMMA_N,
              (N + WMMA_M - 1) / WMMA_M,
              1);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gemm_tensorcore<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    std::printf("Tensor Core GEMM kernel time: %.3f ms\n", kernel_ms);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, float_bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    const float epsilon = 1e-2f;
    for (int row = 0; row < N && ok; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            const float diff = std::fabs(hC[row * N + col] - hC_ref[row * N + col]);
            if (diff > epsilon)
            {
                std::printf("Mismatch at (%d, %d): gpu=%f cpu=%f diff=%f\n",
                            row, col, hC[row * N + col], hC_ref[row * N + col], diff);
                ok = false;
                break;
            }
        }
    }

    if (ok)
    {
        std::printf("Tensor Core GEMM result matches CPU reference\n");
    }
    else
    {
        std::printf("Tensor Core GEMM result does not match CPU reference\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return ok ? 0 : 1;
}
