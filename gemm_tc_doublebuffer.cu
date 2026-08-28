#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
// tensor core v3：使用简单的sharedmemory搬运数据+wmma（load+mma+store）实现矩阵乘法
// 但是这次是4个warp合作处理一个blocktile，使用更多的sharedmemory来减少global memory访问次数
// 双缓冲

using namespace nvcuda::wmma;

// BLOCK大小决定多少个线程合作处理这件事，以32各位一组（warp）
// blocktile的大小根据需要几个warp决定
constexpr int BLOCK_SIZE_M = 64;
constexpr int BLOCK_SIZE_N = 32;
constexpr int BLOCK_SIZE_K = 16;

// TILE_SIZE决定进sharedmemory的块大小
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
// constexpr int TILE_K = 32;

// WMMA_SIZE决定fragment的大小，需要和tensorcore处理能力对齐
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#define CUDA_CHECK(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess)                                          \
        {                                                                \
            std::printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                        cudaGetErrorString(err));                        \
            std::exit(EXIT_FAILURE);                                     \
        }                                                                \
    } while (0)

static float rand_float()
{
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
}

__global__ void gemm_tensorcore(
    const half *A,
    const half *B,
    float *C,
    int N)
{

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int num_thread = blockDim.x * blockDim.y;
    const int warp_id = tid / warpSize;
    const int warp_m = warp_id / (BLOCK_SIZE_N / TILE_N);
    const int warp_n = warp_id % (BLOCK_SIZE_N / TILE_N);

    const int block_m =
        blockIdx.y * BLOCK_SIZE_M;

    const int block_n =
        blockIdx.x * BLOCK_SIZE_N;

    __shared__ half sA[2][BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ half sB[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
    __shared__ float sC[BLOCK_SIZE_M][BLOCK_SIZE_N];

    auto pipeline = cuda::make_pipeline();

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag[TILE_M / WMMA_M];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag[TILE_N / WMMA_N];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[TILE_M / WMMA_M][TILE_N / WMMA_N];

    for (int i = 0; i < TILE_M / WMMA_M; i++)
    {
        for (int j = 0; j < TILE_N / WMMA_N; j++)
        {
            fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // global M ->shared M
    int buffer = 0;
    for (int idx = tid; idx < BLOCK_SIZE_M * BLOCK_SIZE_K; idx += num_thread)
    {
        const int r = idx / BLOCK_SIZE_K;
        const int c = idx % BLOCK_SIZE_K;
        const int g_row = block_m + r;
        const int g_col = c;
        if(g_row<N && g_col<N)
        {
        cuda::memcpy_async(&sA[buffer][r][c], &A[g_row * N + g_col], sizeof(half), pipeline);
        }
        else
        {
            sA[buffer][r][c] = __float2half(0.0f);
        }
    }

    for (int idx = tid; idx < BLOCK_SIZE_K * BLOCK_SIZE_N; idx += num_thread)
    {
        const int r = idx / BLOCK_SIZE_N;
        const int c = idx % BLOCK_SIZE_N;
        const int g_row = r;
        const int g_col = block_n + c;

        if (g_row < N && g_col < N)
        {
            cuda::memcpy_async(&sB[buffer][r][c], &B[g_row * N + g_col], sizeof(half), pipeline);
        }
        else
        {
            sB[buffer][r][c] = __float2half(0.0f);
        }
    }

   pipeline.producer_commit();

    pipeline.consumer_wait();

    __syncthreads();

#pragma unroll
    for (int tile_k = 0; tile_k < N; tile_k += BLOCK_SIZE_K)
    {
        int next = buffer ^ 1;
        if (tile_k + BLOCK_SIZE_K < N)
        {
        for (int idx = tid; idx < BLOCK_SIZE_M * BLOCK_SIZE_K; idx += num_thread)
        {
            const int r = idx / BLOCK_SIZE_K;
            const int c = idx % BLOCK_SIZE_K;
            const int g_row = block_m + r;
            const int g_col = tile_k + c + BLOCK_SIZE_K;
            if(g_row<N && g_col<N){
                cuda::memcpy_async(&sA[next][r][c], &A[g_row * N + g_col], sizeof(half), pipeline);
            }else{
                sA[next][r][c] = __float2half(0.0f);
            }
            
        }

        for (int idx = tid; idx < BLOCK_SIZE_K * BLOCK_SIZE_N; idx += num_thread)
        {
            const int r = idx / BLOCK_SIZE_N;
            const int c = idx % BLOCK_SIZE_N;
            const int g_row = tile_k + r + BLOCK_SIZE_K;
            const int g_col = block_n + c;
            if(g_row<N && g_col<N){
               cuda::memcpy_async(&sB[next][r][c], &B[g_row * N + g_col], sizeof(half), pipeline);
            }else{
                sB[next][r][c] = __float2half(0.0f);
            }
            
        }
        pipeline.producer_commit();

    }

        // shared M -> register(fragment,方便后续tensorcore使用)
        for (int k = 0; k < BLOCK_SIZE_K; k += WMMA_K)
        {
            for (int i = 0; i < TILE_M / WMMA_M; i++)
            {
                load_matrix_sync(a_frag[i], &sA[buffer][warp_m * TILE_M + i * WMMA_M][k], BLOCK_SIZE_K);
            }
            for (int i = 0; i < TILE_N / WMMA_N; i++)
            {
                load_matrix_sync(b_frag[i], &sB[buffer][k][warp_n * TILE_N + i * WMMA_N], BLOCK_SIZE_N);
            }

            for (int i = 0; i < TILE_M / WMMA_M; i++)
            {
                for (int j = 0; j < TILE_N / WMMA_N; j++)
                {
                    mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

       pipeline.consumer_release();
        if(tile_k+BLOCK_SIZE_K<N)
    {

        pipeline.consumer_wait();

        __syncthreads();

        buffer=next;

    }
    }
    for (int i = 0; i < TILE_M / WMMA_M; i++)
    {
        for (int j = 0; j < TILE_N / WMMA_N; j++)
        {

            store_matrix_sync(
                &sC[warp_m * TILE_M + i * WMMA_M][warp_n * TILE_N + j * WMMA_N],
                c_frag[i][j], BLOCK_SIZE_N, mem_row_major);
        }
    }

    __syncthreads();

    for (int idx = tid; idx < BLOCK_SIZE_M * BLOCK_SIZE_N; idx += num_thread)
    {
        const int r = idx / BLOCK_SIZE_N;
        const int c = idx % BLOCK_SIZE_N;
        const int g_row = block_m + r;
        const int g_col = block_n + c;
        if (g_row < N && g_col < N)
        {
            C[g_row * N + g_col] = sC[r][c];
        }
    }
}

int main(int argc, char **argv)
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int N = (argc > 1) ? std::atoi(argv[1]) : 4096;
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

    // for (int row = 0; row < N; ++row)
    // {
    //     for (int col = 0; col < N; ++col)
    //     {
    //         float sum = 0.0f;
    //         for (int k = 0; k < N; ++k)
    //         {
    //             sum += __half2float(hA_half[row*N+k]) * __half2float(hB_half[k*N+col]);
    //         }
    //         hC_ref[row * N + col] = sum;
    //     }
    // }

    half *dA = nullptr;
    half *dB = nullptr;
    float *dC = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, half_bytes));
    CUDA_CHECK(cudaMalloc(&dB, half_bytes));
    CUDA_CHECK(cudaMalloc(&dC, float_bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA_half.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB_half.data(), half_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, float_bytes));

    dim3 block(32, 8);
    dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
              (N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

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
    // for (int row = 0; row < N && ok; ++row)
    // {
    //     for (int col = 0; col < N; ++col)
    //     {
    //         const float diff = std::fabs(hC[row * N + col] - hC_ref[row * N + col]);
    //         if (diff > epsilon)
    //         {
    //             std::printf("Mismatch at (%d, %d): gpu=%f cpu=%f diff=%f\n",
    //                         row, col, hC[row * N + col], hC_ref[row * N + col], diff);
    //             ok = false;
    //             break;
    //         }
    //     }
    // }

    // if (ok)
    // {
    //     std::printf("Tensor Core GEMM result matches CPU reference\n");
    // }
    // else
    // {
    //     std::printf("Tensor Core GEMM result does not match CPU reference\n");
    // }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return ok ? 0 : 1;
}
