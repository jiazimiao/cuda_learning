// Build (CUDA 12.0 or newer):
//   nvcc -O3 -arch=sm_90a -o wgmma_m64n64k16 wgmma_m64n64k16.cu
//
// This is a deliberately small, correctness-oriented WGMMA GEMM.  It uses one
// 128-thread warpgroup per CTA and one m64n64k16 WGMMA per K tile.  It is not
// intended to rival a pipelined CUTLASS kernel; the wait_group<0> inside the K
// loop makes the control flow and shared-memory reuse unambiguous.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int M = 4096;
constexpr int N = 4096;
constexpr int K = 4096;
constexpr int WM = 64;
constexpr int WN = 64;
constexpr int WK = 16;
constexpr int WARP_GROUP_THREADS = 128;

#ifndef WGMMA_USE_32B_SWIZZLE
#define WGMMA_USE_32B_SWIZZLE 1
#endif

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = (call);                                               \
    if (status_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(status_));                                \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

// WGMMA's descriptor stores these three quantities as (bytes >> 4):
// [13:0] start address, [29:16] leading (K) core offset,
// [45:32] stride (M/N) core offset.
__device__ __forceinline__ uint64_t make_wgmma_desc(const void* smem,
                                                     uint32_t leading_bytes,
                                                     uint32_t stride_bytes) {
  const uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint64_t desc = (static_cast<uint64_t>((address       & 0x3ffffu) >> 4)      ) |
                  (static_cast<uint64_t>((leading_bytes & 0x3ffffu) >> 4) << 16) |
                  (static_cast<uint64_t>((stride_bytes  & 0x3ffffu) >> 4) << 32);
#if WGMMA_USE_32B_SWIZZLE
  // ===== BEGIN 32B-SWIZZLE DIFFERENCE: descriptor layout type =====
  desc |= 3ull << 62;  // SM90 WGMMA B32 swizzle
  // ===== END 32B-SWIZZLE DIFFERENCE =====
#endif
  return desc;
}

// ===== BEGIN 32B-SWIZZLE DIFFERENCE: logical -> physical shared address =====
// B32 (Swizzle<1,4,3>), in half-element units, for a 256B-aligned tile base.
__device__ __forceinline__ int swizzle_32b_half_index(int logical_index) {
  return logical_index ^ ((logical_index & 0x40) >> 4);
}
// ===== END 32B-SWIZZLE DIFFERENCE =====

// No-swizzle WGMMA storage is not a normal flat row/column-major array: it is
// a contiguous sequence of 8x8 cores. A cores are row-major; B cores are
// column-major.  Core order is K-major, then M (for A) or N (for B).
__device__ __forceinline__ int noswizzle_a_index(int m, int k) {
  return ((m >> 3) * 2 + (k >> 3)) * 64 + (m & 7) * 8 + (k & 7);
}

__device__ __forceinline__ int noswizzle_b_index(int k, int n) {
  return ((n >> 3) * 2 + (k >> 3)) * 64 + (n & 7) * 8 + (k & 7);
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_group_0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// Each lane owns N/2 = 32 FP32 registers for m64n64k16.  The accumulator is
// read-write (+f): the instruction implements D = A*B + D (scale-d == 1).
__device__ __forceinline__ void wgmma_m64n64k16_f32_f16_f16(
    float (&d)[32], uint64_t desc_a, uint64_t desc_b) {
  constexpr int scale_d = 1;
  asm volatile(
      "{\n"
      "  .reg .pred p;\n"
      "  setp.ne.b32 p, %34, 0;\n"
      "  wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, "
      " %8, %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
        "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
        "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d)
      : "memory");
}

__global__ __launch_bounds__(WARP_GROUP_THREADS)
void wgmma_gemm_4096(const half* __restrict__ A,
                     const half* __restrict__ B,
                     float* __restrict__ C) {
  // Logical A is row-major [64][16] and logical B is column-major [16][64].
#if WGMMA_USE_32B_SWIZZLE
  // ===== BEGIN 32B-SWIZZLE DIFFERENCE: 256B base alignment =====
  __shared__ __align__(256) half sA[WM * WK];
  __shared__ __align__(256) half sB[WN * WK];
  // ===== END 32B-SWIZZLE DIFFERENCE =====
#else
  // Physical shared-memory storage is the no-swizzle 8x8-core layout.
  __shared__ __align__(16) half sA[WM * WK];
  __shared__ __align__(16) half sB[WN * WK];
#endif

  const int tid = threadIdx.x;
  const int block_m = blockIdx.y * WM;
  const int block_n = blockIdx.x * WN;
  float d[32] = {};

  // All four contiguous warps execute every WGMMA instruction identically.
  wgmma_fence();

  #pragma unroll 1
  for (int k0 = 0; k0 < K; k0 += WK) {
    // 128 threads cooperatively stage both 2 KiB operands.
    #pragma unroll
    for (int e = tid; e < WM * WK; e += WARP_GROUP_THREADS) {
      const int row = e / WK;
      const int col = e % WK;
#if WGMMA_USE_32B_SWIZZLE
      // ===== BEGIN 32B-SWIZZLE DIFFERENCE: swizzled shared stores =====
      sA[swizzle_32b_half_index(e)] = A[(block_m + row) * K + k0 + col];
      sB[swizzle_32b_half_index(e)] = B[(k0 + col) * N + block_n + row];
      // ===== END 32B-SWIZZLE DIFFERENCE =====
#else
      sA[noswizzle_a_index(row, col)] = A[(block_m + row) * K + k0 + col];
      sB[noswizzle_b_index(col, row)] = B[(k0 + col) * N + block_n + row];
#endif
    }
    __syncthreads();

    // Every writer publishes its generic shared-memory stores to WGMMA's async
    // proxy before the warpgroup consumes sA/sB.
    fence_proxy_async_shared_cta();
    // A proxy fence is per-thread.  This CTA rendezvous ensures every writer's
    // fence has happened before any warp starts the warpgroup MMA.
    __syncthreads();

#if WGMMA_USE_32B_SWIZZLE
    // ===== BEGIN 32B-SWIZZLE DIFFERENCE: leading offset is one 16B K slice =====
    const uint64_t desc_a = make_wgmma_desc(sA, 16, 256);
    const uint64_t desc_b = make_wgmma_desc(sB, 16, 256);
    // ===== END 32B-SWIZZLE DIFFERENCE =====
#else
    // An 8x8 FP16 core is 128B. K has two core matrices, so adjacent M/N
    // cores are 2 * 128 = 256B apart.
    const uint64_t desc_a = make_wgmma_desc(sA, 128, 256);
    const uint64_t desc_b = make_wgmma_desc(sB, 128, 256);
#endif
    wgmma_m64n64k16_f32_f16_f16(d, desc_a, desc_b);
    wgmma_commit_group();
    wgmma_wait_group_0();  // d and sA/sB are not accessed before completion.
    __syncthreads();       // safe to overwrite the shared tiles next iteration.
  }

  // PTX's m64nNk16 FP32 accumulator layout for a 128-thread warpgroup:
  // each warp owns 16 rows; lane bits [4:2] select a row in each 8-row half,
  // lane bits [1:0] select a pair of columns, and each four registers cover
  // two rows x two columns for one 8-column group.
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row0 = warp * 16 + (lane >> 2);
  const int col2 = (lane & 3) * 2;
  #pragma unroll
  for (int group = 0; group < 8; ++group) {
    const int c = block_n + group * 8 + col2;
    const int r = group * 4;
    C[(block_m + row0)     * N + c]     = d[r + 0];
    C[(block_m + row0)     * N + c + 1] = d[r + 1];
    C[(block_m + row0 + 8) * N + c]     = d[r + 2];
    C[(block_m + row0 + 8) * N + c + 1] = d[r + 3];
  }
}

static float host_a(int row, int col) {
  return static_cast<float>(((row * 17 + col * 13) % 7) - 3) * 0.125f;
}
static float host_b(int row, int col) {
  return static_cast<float>(((row * 19 + col * 11) % 7) - 3) * 0.125f;
}

static float reference_element(int row, int col) {
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) sum += host_a(row, k) * host_b(k, col);
  return sum;
}

int main() {
  int device = 0, major = 0, minor = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major != 9) {
    std::fprintf(stderr, "This program requires Hopper SM90a; found compute capability %d.%d.\n",
                 major, minor);
    return EXIT_FAILURE;
  }

  const size_t ab_bytes = static_cast<size_t>(M) * K * sizeof(half);
  const size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);
  std::vector<half> hA(static_cast<size_t>(M) * K);
  std::vector<half> hB(static_cast<size_t>(K) * N);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) hA[static_cast<size_t>(i) * K + k] = __float2half_rn(host_a(i, k));
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j) hB[static_cast<size_t>(k) * N + j] = __float2half_rn(host_b(k, j));

  half *dA = nullptr, *dB = nullptr;
  float* dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, ab_bytes));
  CUDA_CHECK(cudaMalloc(&dB, ab_bytes));
  CUDA_CHECK(cudaMalloc(&dC, c_bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), ab_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), ab_bytes, cudaMemcpyHostToDevice));

  const dim3 block(WARP_GROUP_THREADS);
  const dim3 grid(N / WN, M / WM);
  for (int i = 0; i < 5; ++i) wgmma_gemm_4096<<<grid, block>>>(dA, dB, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  constexpr int kIters = 20;
  CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < kIters; ++i) wgmma_gemm_4096<<<grid, block>>>(dA, dB, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
  milliseconds /= kIters;

  std::vector<float> hC(static_cast<size_t>(M) * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, c_bytes, cudaMemcpyDeviceToHost));
  float max_abs_error = 0.0f;
  // The inputs are binary-exact multiples of 1/8, so these FP32 sums are exact
  // at this size.  Check a spread of output elements without an O(N^3) host GEMM.
  for (int sample = 0; sample < 256; ++sample) {
    const int i = (sample * 997) & (M - 1);
    const int j = (sample * 619) & (N - 1);
    max_abs_error = std::max(max_abs_error,
        std::abs(hC[static_cast<size_t>(i) * N + j] - reference_element(i, j)));
  }
  const double tflops = (2.0 * M * N * K) / (static_cast<double>(milliseconds) * 1.0e9);
  std::printf("WGMMA m64n64k16 FP16xFP16->FP32, M=N=K=4096\n");
  std::printf("GEMM duration: %.3f ms, throughput: %.2f TFLOP/s\n", milliseconds, tflops);
  std::printf("Validation (256 CPU-reference samples): max abs error = %.8g %s\n",
              max_abs_error, max_abs_error == 0.0f ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return max_abs_error == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}
