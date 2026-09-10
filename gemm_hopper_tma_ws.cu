// Warp-specialized Hopper GEMM learning version. Original files preserved.
// nvcc -O3 -std=c++17 -lineinfo -gencode arch=compute_90a,code=sm_90a \
//   wgmma_m64n64k16_tma_ws.cu -lcuda -o gemm_ws
// Target: CUDA 13.1, H100 sm_90a. No architecture fallback.
// Search "WS CHANGE" for changes from wgmma_m64n64k16_tma_cpp.cu.
// Dedicated producer warp + one aligned consumer warpgroup, 2-stage pipeline.
// No setmaxnreg / dynamic register redistribution in this teaching version.
// Tensor maps, SW128 descriptors, WGMMA arithmetic and validation unchanged.
#include <cuda/barrier> // [CUDA API CHANGE 1] Typed transaction barrier.
#include <cuda/ptx>     // Official wrappers for 2D tensor copy and proxy fence.
#include <cuda.h>                 // [TMA CHANGE 1] Host tensor-map encoder
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
constexpr int M=4096, N=4096, K=4096;
constexpr int WM=64, WN=64, WK=16, BK=64, STAGES=2;
constexpr int WARP_GROUP_THREADS=128;
// [WS CHANGE 1] Consumers first: preserves WGMMA warpgroup alignment/mapping.
constexpr int PRODUCER_THREADS = 32;
constexpr int PRODUCER_LEADER = WARP_GROUP_THREADS; // thread 128, warp 4 lane 0
constexpr int BLOCK_THREADS = WARP_GROUP_THREADS + PRODUCER_THREADS; // 160
static_assert(BLOCK_THREADS == 160 && WARP_GROUP_THREADS == 128);
static_assert(K % BK == 0 && BK % WK == 0 && STAGES == 2);
constexpr int TILE_ELEMENTS=64*64;
constexpr int TX_BYTES=2*TILE_ELEMENTS*sizeof(half); // A+B = 16384 bytes
#define CUDA_CHECK(call) do { auto e=(call); if(e!=cudaSuccess) { \
  std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  std::exit(EXIT_FAILURE); } } while(0)
#define DRIVER_CHECK(call) do { CUresult e=(call); if(e!=CUDA_SUCCESS) { \
  const char* msg=nullptr; cuGetErrorString(e,&msg); \
  std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,msg?msg:"driver error"); \
  std::exit(EXIT_FAILURE); } } while(0)

static CUtensorMap encode_map(half* ptr, uint64_t inner, uint64_t outer) {
  alignas(64) CUtensorMap map{};
  const cuuint64_t dims[2]={inner,outer};
  const cuuint64_t strides[1]={inner*sizeof(half)};
  const cuuint32_t box[2]={64,64}, element_strides[2]={1,1};
  DRIVER_CHECK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
      2, ptr, dims, strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}
__device__ __forceinline__ uint32_t shared_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// [TMA CHANGE 2] SW128 WGMMA descriptor. All stage bases align to 1024B.
// A: K-major, 64 FP16 per row, SBO=8*128=1024B, LBO assumed 16B.
// B: N-major, 64 FP16 per K row, SBO=8*128=1024B.
// B's LBO=8192B describes the hypothetical next 64-column atom (unused N=64).
// k16 slices: A offset=k16*2 bytes, B offset=k16*128 bytes.
// All slices have base-offset=0: A stays within the first 128B row;
// B moves by 2048B, a multiple of the 1024B swizzle pattern.
__device__ __forceinline__ uint64_t make_desc(const half* p, uint32_t lbo) {
  return uint64_t((shared_addr(p)&0x3ffffu)>>4)
      | (uint64_t(lbo>>4)<<16) | (uint64_t(1024>>4)<<32)
      | (uint64_t(1)<<62); // WGMMA 1=SW128 (not the tensor-map enum value!)
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait_group_0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
}
// One producer arrives at each full barrier; consumers wait by parity.
using TmaBarrier = cuda::barrier<cuda::thread_scope_block>;

// Issue A+B for one pipeline slot. [WS CHANGE] Called ONLY by tid=PRODUCER_LEADER.
// Coordinates are fastest-dimension-first: A={k,m}, B={n,k}.
// The tensor maps, not this function, specify tile size and SW128 layout.
__device__ __forceinline__ void load_stage(
    const CUtensorMap* a, const CUtensorMap* b, half* sa, half* sb,
    TmaBarrier& ready, int k0, int m0, int n0) {
  // One software arrival + 16384 expected bytes in the SAME phase.
  // Register bytes before issuing copies, so completion cannot race ahead.
  // The returned token is intentionally unused: consumers use wait_parity().
  (void)cuda::device::barrier_arrive_tx(ready, 1, TX_BYTES);

  // Obtain the actual hardware barrier pointer. Do not reinterpret_cast it,
  // and do not manually convert these C++ pointers to shared-address integers.
  auto* completion = cuda::device::barrier_native_handle(ready);
  const int32_t coordsA[2] = {k0, m0};
  const int32_t coordsB[2] = {n0, k0};

  cuda::ptx::cp_async_bulk_tensor(
      cuda::ptx::space_cluster, cuda::ptx::space_global,
      sa, a, coordsA, completion);
  cuda::ptx::cp_async_bulk_tensor(
      cuda::ptx::space_cluster, cuda::ptx::space_global,
      sb, b, coordsB, completion);
  // Returning means "issued", NOT "data ready".
}

// [TMA CHANGE 4] transA=0 (K-major); transB=1 (N-major).
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
      "%32, %33, p, 1, 1, 0, 1;\n"
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


// [WS CHANGE 2] 5 warps: warps 0..3 compute; warp 4 produces.
// The consumer must remain an aligned group of four consecutive warps.
// All 160 threads reach the two block-wide barriers OUTSIDE the pipeline.
__global__ __launch_bounds__(BLOCK_THREADS)
void wgmma_gemm_ws(const __grid_constant__ CUtensorMap mapA,
                   const __grid_constant__ CUtensorMap mapB, float* C) {
  __shared__ __align__(1024) half sA[STAGES][TILE_ELEMENTS];
  __shared__ __align__(1024) half sB[STAGES][TILE_ELEMENTS];

  // [WS CHANGE 3] Two directions of notification, independently per slot.
  // full:  one producer arrival + A/B transaction completion -> may read.
  // empty: all 128 consumer arrivals after WGMMA wait -> may overwrite.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ TmaBarrier full[STAGES];
  __shared__ TmaBarrier empty[STAGES];

  const int tid = threadIdx.x;
  const int block_m = blockIdx.y * WM;
  const int block_n = blockIdx.x * WN;

  if (tid == PRODUCER_LEADER) {
    #pragma unroll
    for (int slot = 0; slot < STAGES; ++slot) {
      init(&full[slot], 1);
      init(&empty[slot], WARP_GROUP_THREADS);
    }
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }
  __syncthreads(); // One-time publication to every thread, not in the loop.

  // [WS CHANGE 4] Producer loop runs independently of the consumer loop.
  // Only lane 0 of the dedicated producer warp issues TMA.
  // The other 31 producer lanes have no per-tile work in this minimal design.
  if (tid == PRODUCER_LEADER) {
    #pragma unroll 1
    for (int t = 0; t < K / BK; ++t) {
      const int slot = t % STAGES;
      const int generation = t / STAGES;
      if (generation != 0) {
        // First use is free. Later uses wait for the PREVIOUS generation
        // to be consumed. No artificial initial arrival is needed.
        empty[slot].wait_parity((generation - 1) & 1);
      }
      load_stage(&mapA, &mapB, sA[slot], sB[slot], full[slot],
                 t * BK, block_m, block_n);
    }
  } else if (tid < WARP_GROUP_THREADS) {
    // [WS CHANGE 5] Only the 128 consumers own/use accumulator fragments.
    // Their IDs are still 0..127: original WGMMA fragment mapping unchanged.
    float d[32] = {};
    wgmma_fence();

    #pragma unroll 1
    for (int t = 0; t < K / BK; ++t) {
      const int slot = t % STAGES;
      const int generation = t / STAGES;
      full[slot].wait_parity(generation & 1);

      #pragma unroll
      for (int kk = 0; kk < BK; kk += WK) {
        const uint64_t da = make_desc(sA[slot] + kk, 16);
        const uint64_t db = make_desc(sB[slot] + kk * WN, 8192);
        wgmma_m64n64k16_f32_f16_f16(d, da, db);
      }
      wgmma_commit_group();
      wgmma_wait_group_0();

      // [WS CHANGE 6] EVERY consumer releases only after its WGMMA wait.
      // empty was initialized to 128, so the producer cannot overwrite
      // until all consumers have finished using the slot.
      // Do NOT replace this with __syncthreads(): producer runs another loop.
      (void)empty[slot].arrive();
    }

    // Original accumulator-to-output mapping, consumer IDs unchanged.
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

  // [WS CHANGE 7] Join after both loops, including inactive producer lanes.
  // All TMA loads have been consumed and all WGMMA groups have completed.
  __syncthreads();
  if (tid == PRODUCER_LEADER) {
    #pragma unroll
    for (int slot = 0; slot < STAGES; ++slot) {
      full[slot].~TmaBarrier();
      empty[slot].~TmaBarrier();
    }
  }
}

static float host_a(int row, int col) {
  return static_cast<float>(((row * 17 + col * 13) % 7) - 3) * 0.125f;
}
static float host_b(int row, int col) {
  return static_cast<float>(((row * 19 + col * 11) % 7) - 3) * 0.125f;
}

static float reference_element(int row, int col, bool identity_a, bool identity_b) {
  if (identity_a) return host_b(row, col);  // A is I, so C = B.
  if (identity_b) return host_a(row, col);  // B is I, so C = A.
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) sum += host_a(row, k) * host_b(k, col);
  return sum;
}

int main(int argc, char** argv) {
  std::printf("Layout revision: WS-1P4C-SW128-K64-2stage-v1\n");
  DRIVER_CHECK(cuInit(0));
  const bool identity_a = argc == 2 && std::strcmp(argv[1], "--identity-a") == 0;
  const bool identity_b = argc == 2 && std::strcmp(argv[1], "--identity-b") == 0;
  const bool identity_a_n = argc == 2 && std::strcmp(argv[1], "--identity-a-n") == 0;
  const bool identity_a_k = argc == 2 && std::strcmp(argv[1], "--identity-a-k") == 0;
  const bool any_identity_a = identity_a || identity_a_n || identity_a_k;
  if (argc > 1 && !any_identity_a && !identity_b) {
    std::fprintf(stderr,
                 "Usage: %s [--identity-a | --identity-b | --identity-a-n | --identity-a-k]\n",
                 argv[0]);
    return EXIT_FAILURE;
  }
  int device = 0, major = 0, minor = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (major != 9 || minor != 0) {
    std::fprintf(stderr, "This program requires Hopper SM90a; found compute capability %d.%d.\n",
                 major, minor);
    return EXIT_FAILURE;
  }

  const size_t ab_bytes = static_cast<size_t>(M) * K * sizeof(half);
  const size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);
  std::vector<half> hA(static_cast<size_t>(M) * K);
  std::vector<half> hB(static_cast<size_t>(K) * N);
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      hA[static_cast<size_t>(i) * K + k] = __float2half_rn(any_identity_a ? (i == k ? 1.0f : 0.0f) : host_a(i, k));
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      hB[static_cast<size_t>(k) * N + j] = __float2half_rn(
          identity_b ? (k == j ? 1.0f : 0.0f) :
          identity_a_n ? static_cast<float>(j & 63) :
          identity_a_k ? static_cast<float>(k & 63) : host_b(k, j));

  half *dA = nullptr, *dB = nullptr;
  float* dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, ab_bytes));
  CUDA_CHECK(cudaMalloc(&dB, ab_bytes));
  CUDA_CHECK(cudaMalloc(&dC, c_bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), ab_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), ab_bytes, cudaMemcpyHostToDevice));

  // [TMA CHANGE 5] Tensor-map dimensions are fastest-first; no transpose.
  // A coordinates {k,m}, B coordinates {n,k}. Each box is 64x64 FP16.
  alignas(64) CUtensorMap mapA = encode_map(dA, K, M);
  alignas(64) CUtensorMap mapB = encode_map(dB, N, K);
  const dim3 block(BLOCK_THREADS); // [WS CHANGE 8] Launch all 160 threads.
  const dim3 grid(N / WN, M / WM);
  for (int i = 0; i < 5; ++i) wgmma_gemm_ws<<<grid, block>>>(mapA, mapB, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t begin, end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  constexpr int kIters = 20;
  CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < kIters; ++i) wgmma_gemm_ws<<<grid, block>>>(mapA, mapB, dC);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
  milliseconds /= kIters;

  std::vector<float> hC(static_cast<size_t>(M) * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, c_bytes, cudaMemcpyDeviceToHost));
  float max_abs_error = 0.0f;
  int mismatches_reported = 0;
  // [TMA CHANGE 7] Validate every output. The existing deterministic A/B
  // generators have period 7 in each coordinate. Cache the 49 independent
  // CPU dot products (each still sums all K terms), then check all M*N values.
  // This shortcut is valid only for these generators; replace it if inputs change.
  float reference[7][7] = {};
  if (!any_identity_a && !identity_b)
    for (int i=0;i<7;++i)
      for (int j=0;j<7;++j)
        reference[i][j]=reference_element(i,j,false,false);
  for (size_t sample = 0; sample < static_cast<size_t>(M)*N; ++sample) {
    const int i = static_cast<int>(sample/N);
    const int j = static_cast<int>(sample%N);
    const float expected = identity_a_n ? static_cast<float>(j & 63) :
                           identity_a_k ? static_cast<float>(i & 63) :
                           identity_a ? host_b(i,j) :
                           identity_b ? host_a(i,j) : reference[i%7][j%7];
    const float actual = hC[static_cast<size_t>(i) * N + j];
    const float abs_error = std::abs(actual - expected);
    max_abs_error = std::isfinite(actual) ? std::max(max_abs_error, abs_error) : INFINITY;
    if (abs_error != 0.0f && mismatches_reported < 8) {
      std::printf("  mismatch C[%d,%d]: GPU=%g CPU=%g abs_err=%g\n",
                  i, j, actual, expected, abs_error);
      ++mismatches_reported;
    }
  }
  const double tflops = (2.0 * M * N * K) / (static_cast<double>(milliseconds) * 1.0e9);
  std::printf("WGMMA m64n64k16 FP16xFP16->FP32, M=N=K=4096; TMA SW128, BK=64, stages=2; WS: 1 producer warp + 4 consumer warps\n");
  std::printf("GEMM duration: %.3f ms, throughput: %.2f TFLOP/s\n", milliseconds, tflops);
  std::printf("Validation (all 16777216 outputs, CPU reference): max abs error = %.8g %s\n",
              max_abs_error, max_abs_error == 0.0f ? "PASS" : "FAIL");

  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return max_abs_error == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}
