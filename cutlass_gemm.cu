#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t err = (call);                                                            \
    if (err != cudaSuccess) {                                                            \
      std::printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                     \
                  cudaGetErrorString(err));                                              \
      std::exit(EXIT_FAILURE);                                                           \
    }                                                                                    \
  } while (0)

static float rand_float() {
  return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
}

int main(int argc, char** argv) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));

  int M = (argc > 1) ? std::atoi(argv[1]) : 1024;
  int N = (argc > 2) ? std::atoi(argv[2]) : 1024;
  int K = (argc > 3) ? std::atoi(argv[3]) : 1024;

  if (M <= 0 || N <= 0 || K <= 0) {
    std::printf("Invalid shape M=%d N=%d K=%d\n", M, N, K);
    return 1;
  }

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = float;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator>;

  size_t elemsA = static_cast<size_t>(M) * K;
  size_t elemsB = static_cast<size_t>(K) * N;
  size_t elemsC = static_cast<size_t>(M) * N;

  std::vector<float> hA_fp32(elemsA);
  std::vector<float> hB_fp32(elemsB);
  std::vector<ElementA> hA(elemsA);
  std::vector<ElementB> hB_colmajor(elemsB);
  std::vector<ElementC> hC(elemsC, 0.0f);
  std::vector<ElementC> hC_ref(elemsC, 0.0f);

  for (size_t i = 0; i < elemsA; ++i) {
    hA_fp32[i] = rand_float();
    hA[i] = static_cast<ElementA>(hA_fp32[i]);
  }

  // B is generated in row-major float first, then packed into column-major for CUTLASS.
  for (int r = 0; r < K; ++r) {
    for (int c = 0; c < N; ++c) {
      float v = rand_float();
      hB_fp32[static_cast<size_t>(r) * N + c] = v;
      hB_colmajor[static_cast<size_t>(c) * K + r] = static_cast<ElementB>(v);
    }
  }

  // CPU reference GEMM in FP32.
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += hA_fp32[static_cast<size_t>(i) * K + k] *
               hB_fp32[static_cast<size_t>(k) * N + j];
      }
      hC_ref[static_cast<size_t>(i) * N + j] = sum;
    }
  }

  ElementA* dA = nullptr;
  ElementB* dB = nullptr;
  ElementC* dC = nullptr;

  CUDA_CHECK(cudaMalloc(&dA, elemsA * sizeof(ElementA)));
  CUDA_CHECK(cudaMalloc(&dB, elemsB * sizeof(ElementB)));
  CUDA_CHECK(cudaMalloc(&dC, elemsC * sizeof(ElementC)));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), elemsA * sizeof(ElementA), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB_colmajor.data(), elemsB * sizeof(ElementB), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, elemsC * sizeof(ElementC)));

  CutlassGemm gemm_op;

  cutlass::gemm::GemmCoord problem_size(M, N, K);
  ElementC alpha = 1.0f;
  ElementC beta = 0.0f;

  typename CutlassGemm::Arguments args(
      problem_size,
      {dA, K},   // A row-major ld = K
      {dB, K},   // B col-major ld = K
      {dC, N},   // C row-major ld = N
      {dC, N},   // D row-major ld = N
      {alpha, beta});

  cutlass::Status can_implement = gemm_op.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::printf("CUTLASS cannot implement this GEMM config: %s\n",
                cutlassGetStatusString(can_implement));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 1;
  }

  cutlass::Status init_status = gemm_op.initialize(args);
  if (init_status != cutlass::Status::kSuccess) {
    std::printf("CUTLASS initialize failed: %s\n", cutlassGetStatusString(init_status));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 1;
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  cutlass::Status run_status = gemm_op();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  if (run_status != cutlass::Status::kSuccess) {
    std::printf("CUTLASS run failed: %s\n", cutlassGetStatusString(run_status));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 1;
  }

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::printf("CUTLASS GEMM kernel time: %.3f ms\n", ms);

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, elemsC * sizeof(ElementC), cudaMemcpyDeviceToHost));

  bool ok = true;
  const float eps = 1e-2f;
  for (int i = 0; i < M && ok; ++i) {
    for (int j = 0; j < N; ++j) {
      float got = hC[static_cast<size_t>(i) * N + j];
      float ref = hC_ref[static_cast<size_t>(i) * N + j];
      float diff = std::fabs(got - ref);
      if (diff > eps) {
        std::printf("Mismatch at (%d,%d): got=%f ref=%f diff=%f\n", i, j, got, ref, diff);
        ok = false;
        break;
      }
    }
  }

  if (ok) {
    std::printf("CUTLASS GEMM result matches CPU reference\n");
  } else {
    std::printf("CUTLASS GEMM result does not match CPU reference\n");
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));

  return ok ? 0 : 1;
}
