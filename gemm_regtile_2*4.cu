#include <stdio.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <cuda/cmath>
#include <cuda_runtime.h>
#define N 4096
#define THREAD_TILE 16
#define BM 64
#define BN 32
#define BK 16

__global__ void gemm(
    float *A,
    float *B,
    float *C)
{
    // A block has 16x16 threads and computes a 64x32 C tile.
    // Each thread accumulates a 4x2 C micro-tile in registers.
    __shared__ float tileA[BM][BK];
    __shared__ float tileB[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = blockIdx.y * BM + ty;
    const int row1 = row0 + THREAD_TILE;
    const int row2 = row1 + THREAD_TILE;
    const int row3 = row2 + THREAD_TILE;
    const int col0 = blockIdx.x * BN + tx;
    const int col1 = col0 + THREAD_TILE;

    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    float sum20 = 0.0f;
    float sum21 = 0.0f;
    float sum30 = 0.0f;
    float sum31 = 0.0f;

    for (int tile = 0; tile < (N + BK - 1) / BK; ++tile)
    {
        const int kA = tile * BK + tx;
        const int kB = tile * BK + ty;

        // Each thread loads two A values and two B values.
        // Out-of-range elements are zero-filled for edge tiles.
        tileA[ty][tx] =
            (row0 < N && kA < N) ? A[row0 * N + kA] : 0.0f;
        tileA[ty + THREAD_TILE][tx] =
            (row1 < N && kA < N) ? A[row1 * N + kA] : 0.0f;
        tileA[ty + 2 * THREAD_TILE][tx] =
            (row2 < N && kA < N) ? A[row2 * N + kA] : 0.0f;
        tileA[ty + 3 * THREAD_TILE][tx] =
            (row3 < N && kA < N) ? A[row3 * N + kA] : 0.0f;

        tileB[ty][tx] =
            (kB < N && col0 < N) ? B[kB * N + col0] : 0.0f;
        tileB[ty][tx + THREAD_TILE] =
            (kB < N && col1 < N) ? B[kB * N + col1] : 0.0f;
    

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            const float a0 = tileA[ty][k];
            const float a1 = tileA[ty + THREAD_TILE][k];
            const float a2 = tileA[ty + 2 * THREAD_TILE][k];
            const float a3 = tileA[ty + 3 * THREAD_TILE][k];
            const float b0 = tileB[k][tx];
            const float b1 = tileB[k][tx + THREAD_TILE];


            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
            sum20 += a2 * b0;
            sum21 += a2 * b1;
            sum30 += a3 * b0;
            sum31 += a3 * b1;
            
        }

        __syncthreads();

        
    }
    if (row0 < N && col0 < N) C[row0 * N + col0] = sum00;
    if (row0 < N && col1 < N) C[row0 * N + col1] = sum01;
    if (row1 < N && col0 < N) C[row1 * N + col0] = sum10;
    if (row1 < N && col1 < N) C[row1 * N + col1] = sum11;
    if (row2 < N && col0 < N) C[row2 * N + col0] = sum20;
    if (row2 < N && col1 < N) C[row2 * N + col1] = sum21;
    if (row3 < N && col0 < N) C[row3 * N + col0] = sum30;
    if (row3 < N && col1 < N) C[row3 * N + col1] = sum31;
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecmul(float* A, float* B, float* C,  int M, int K, int L)
{
    for(int i=0; i<M; i++)
    {
        for(int j = 0; j < L; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * L + j];
            }
            C[i * L + j] = sum;
        }
        
    }
    
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=1e-3f)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main()
{

    size_t size = N * N * sizeof(float);
    int vectorLength = N * N;

    float *A;
    float *B;
    float *C;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // CPU memory

    // A=(float*)malloc(size);
    // B=(float*)malloc(size);
    // C=(float*)malloc(size);
    cudaMallocHost(&A, size);
    cudaMallocHost(&B, size);
    cudaMallocHost(&C, size);

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    float *d_A;
    float *d_B;
    float *d_C;

    // GPU memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // CPU -> GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size);

    // define block

    dim3 block(THREAD_TILE, THREAD_TILE);
    dim3 grid((N + BN - 1) / BN,
              (N + BM - 1) / BM);
  

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gemm<<<grid, block>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess)
    {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float kernelMilliseconds = 0.0f;
    cudaEventElapsedTime(&kernelMilliseconds, start, stop);
    printf("GEMM kernel time: %.3f ms\n", kernelMilliseconds);

    serialVecmul(A, B, comparisonResult, N, N, N);

    // GPU -> CPU

    cudaMemcpy(
        C,
        d_C,
        size,
        cudaMemcpyDeviceToHost);

    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("CPU and GPU answers match\n");
    }
    else
    {
        printf("Error - CPU and GPU answers do not match\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free(A);
    // free(B);
    // free(C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}
