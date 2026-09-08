#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

namespace cg = cooperative_groups;

// Hopper GEMM learning kernel, no CUTLASS and no CuTe.
//
// Design:
//   - one CTA = one warpgroup = 128 threads
//   - one CTA computes one 64x64 C tile
//   - fp16 A/B, fp32 accumulation, fp32 C
//   - cuda::memcpy_async + cuda::barrier stages gmem -> smem
//   - inline PTX wgmma.mma_async does smem -> tensor core -> registers
//
// Notes:
//   - This is a learning kernel, not a tuned GEMM.
//   - N must be a multiple of 64, so all tiles are full.
//   - On Hopper+, the cuda::memcpy_async barrier overload may map to TMA
//     cp.async.bulk when alignment and memory-space conditions are satisfied.
//   - WGMMA itself has no CUDA C++ API like WMMA, so inline PTX is used.
//
// Build on H100/H800:
//   nvcc -std=c++17 -O3 -arch=sm_90a \
//        gemm_hopper_no_cutlass_no_cute.cu -o gemm_hopper_no_cutlass
//
// Run:
//   ./gemm_hopper_no_cutlass 256
//   ./gemm_hopper_no_cutlass 4096

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;
constexpr int WGMMA_K = 16;
constexpr int WARP_GROUP_THREADS = 128;
constexpr int STAGES = 2;

constexpr int A_LD_SMEM = BLOCK_K;
constexpr int B_LD_SMEM = BLOCK_N;

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

__device__ __forceinline__ uint32_t smem_u32addr(const void *ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint64_t make_wgmma_desc(const void *smem_ptr,
                                                    int leading_byte_offset,
                                                    int stride_byte_offset)
{
    const uint64_t start = static_cast<uint64_t>(smem_u32addr(smem_ptr) >> 4);
    const uint64_t lbo = static_cast<uint64_t>(leading_byte_offset >> 4);
    const uint64_t sbo = static_cast<uint64_t>(stride_byte_offset >> 4);

    // Unswizzled WGMMA shared-memory descriptor:
    //   bits [0:13]   base address / 16
    //   bits [16:29]  leading dimension byte offset / 16
    //   bits [32:45]  stride dimension byte offset / 16
    return (start & 0x3fffULL) |
           ((lbo & 0x3fffULL) << 16) |
           ((sbo & 0x3fffULL) << 32);
}

__device__ __forceinline__ void wgmma_fence()
{
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group()
{
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_group_0()
{
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}

__device__ __forceinline__ void wgmma_m64n64k16_f16_f32(float (&d)[32],
                                                        uint64_t desc_a,
                                                        uint64_t desc_b,
                                                        int scale_d)
{
    // A and B are both in shared memory. This instruction is warpgroup-scoped:
    // all 128 threads in the CTA execute it together.
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d)
        : "memory");
}

__device__ __forceinline__ void store_wgmma_m64n64(float *C,
                                                   const float (&acc)[32],
                                                   int N,
                                                   int block_m,
                                                   int block_n)
{
    // D-fragment mapping for wgmma.mma_async m64n64 with fp32 accumulators.
    // Each thread owns 32 fp32 values. Across 128 threads that is 4096 values,
    // exactly one 64x64 tile.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int lane_row = lane >> 2;
    const int lane_col_pair = lane & 3;

    #pragma unroll
    for (int col_group = 0; col_group < 8; ++col_group)
    {
        #pragma unroll
        for (int row_half = 0; row_half < 2; ++row_half)
        {
            #pragma unroll
            for (int col_in_pair = 0; col_in_pair < 2; ++col_in_pair)
            {
                const int reg = col_group * 4 + row_half * 2 + col_in_pair;
                const int row = block_m + warp * 16 + lane_row + row_half * 8;
                const int col = block_n + col_group * 8 + lane_col_pair * 2 + col_in_pair;
                C[row * N + col] = acc[reg];
            }
        }
    }
}

__global__ __launch_bounds__(WARP_GROUP_THREADS) void gemm_hopper_kernel(
    const half *A,
    const half *B,
    float *C,
    int N)
{
// #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
//     if (threadIdx.x == 0)
//     {
//         printf("This kernel requires Hopper and must be compiled with -arch=sm_90a.\\n");
//     }
//     return;
// #else
    const int tid = threadIdx.x;
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;
    const cg::thread_block cta = cg::this_thread_block();

    __align__(16) __shared__ half sA[STAGES][BLOCK_M][A_LD_SMEM];
    __align__(16) __shared__ half sB[STAGES][BLOCK_K][B_LD_SMEM];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block> tma_bar[STAGES];

    if (tid == 0)
    {
        init(&tma_bar[0], WARP_GROUP_THREADS);
        init(&tma_bar[1], WARP_GROUP_THREADS);
    }
    __syncthreads();

    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        acc[i] = 0.0f;
    }

    auto load_stage = [&](int stage, int tile_k) {
        // TMA-capable CUDA C++ interface path. Each call is a cooperative
        // aligned gmem->smem bulk copy bound to a shared-memory barrier.
        #pragma unroll
        for (int r = 0; r < BLOCK_M; ++r)
        {
            cuda::memcpy_async(
                cta,
                &sA[stage][r][0],
                &A[(block_m + r) * N + tile_k],
                cuda::aligned_size_t<16>(BLOCK_K * sizeof(half)),
                tma_bar[stage]);
        }

        #pragma unroll
        for (int r = 0; r < BLOCK_K; ++r)
        {
            cuda::memcpy_async(
                cta,
                &sB[stage][r][0],
                &B[(tile_k + r) * N + block_n],
                cuda::aligned_size_t<16>(BLOCK_N * sizeof(half)),
                tma_bar[stage]);
        }
    };

    int stage = 0;
    load_stage(stage, 0);

    for (int tile_k = 0; tile_k < N; tile_k += BLOCK_K)
    {
        const int next_stage = stage ^ 1;
        const int next_tile_k = tile_k + BLOCK_K;

        if (next_tile_k < N)
        {
            load_stage(next_stage, next_tile_k);
        }

        tma_bar[stage].arrive_and_wait();
        __syncthreads();

        // WGMMA reads shared memory through the async proxy. This fence orders
        // the completed shared-memory writes before WGMMA's async-proxy reads.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        wgmma_fence();

        #pragma unroll
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner += WGMMA_K)
        {
            const uint64_t desc_a = make_wgmma_desc(
                &sA[stage][0][k_inner],
                16,
                8 * A_LD_SMEM * static_cast<int>(sizeof(half)));
            const uint64_t desc_b = make_wgmma_desc(
                &sB[stage][k_inner][0],
                16,
                B_LD_SMEM * static_cast<int>(sizeof(half)));

            wgmma_m64n64k16_f16_f32(acc, desc_a, desc_b,
                                    (tile_k == 0 && k_inner == 0) ? 0 : 1);
            wgmma_commit_group();
            wgmma_wait_group_0();
        }

        __syncthreads();
        stage = next_stage;
    }

    wgmma_wait_group_0();
    store_wgmma_m64n64(C, acc, N, block_m, block_n);
// #endif
}

int main(int argc, char **argv)
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
    if (N <= 0 || N % BLOCK_M != 0)
    {
        std::printf("Usage: %s [N], where N is a positive multiple of %d.\n",
                    argv[0], BLOCK_M);
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 9)
    {
        std::printf("This example requires Hopper, compute capability 9.x.\\n");
        return 1;
    }

    const size_t elems = static_cast<size_t>(N) * N;
    const size_t half_bytes = elems * sizeof(half);
    const size_t float_bytes = elems * sizeof(float);

    std::vector<float> hA_float(elems);
    std::vector<float> hB_float(elems);
    std::vector<half> hA(elems);
    std::vector<half> hB(elems);
    std::vector<float> hC(elems, 0.0f);
    std::vector<float> hRef(elems, 0.0f);

    for (size_t i = 0; i < elems; ++i)
    {
        hA_float[i] = rand_float();
        hB_float[i] = rand_float();
        hA[i] = __float2half(hA_float[i]);
        hB[i] = __float2half(hB_float[i]);
    }

    const bool check_result = N <= 512;
    if (check_result)
    {
        for (int row = 0; row < N; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k)
                {
                    sum += __half2float(hA[row * N + k]) *
                           __half2float(hB[k * N + col]);
                }
                hRef[row * N + col] = sum;
            }
        }
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

    dim3 block(WARP_GROUP_THREADS);
    dim3 grid(N / BLOCK_N, N / BLOCK_M);

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    gemm_hopper_kernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    std::printf("No-CUTLASS Hopper GEMM: N=%d time=%.3f ms\n", N, kernel_ms);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, float_bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    if (check_result)
    {
        const float eps = 2e-1f;
        for (int row = 0; row < N && ok; ++row)
        {
            for (int col = 0; col < N; ++col)
            {
                const float diff = std::fabs(hC[row * N + col] - hRef[row * N + col]);
                if (diff > eps)
                {
                    std::printf("Mismatch at (%d,%d): gpu=%f ref=%f diff=%f\n",
                                row, col, hC[row * N + col], hRef[row * N + col], diff);
                    ok = false;
                    break;
                }
            }
        }
        std::printf(ok ? "Result check passed\n" : "Result check failed\n");
    }
    else
    {
        std::printf("Skip CPU reference for N=%d; use N<=512 to validate.\n", N);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return ok ? 0 : 1;
}
