#include <stdio.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <cuda/cmath>
#include <cuda_runtime.h>
#define N 4096
#define TILE_WIDTH 32

__global__ void gemm(
    float *A,
    float *B,
    float *C)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];


    int row =
        blockIdx.y * TILE_WIDTH + threadIdx.y;

    int col =
        blockIdx.x * TILE_WIDTH + threadIdx.x;




    for(int tile = 0;tile<(N+TILE_WIDTH-1)/TILE_WIDTH;tile++)
    {
        if(row<N && tile*TILE_WIDTH+threadIdx.x<N)
        {
            tileA[threadIdx.y][threadIdx.x] = A[row*N + tile*TILE_WIDTH + threadIdx.x];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(col<N && tile*TILE_WIDTH+threadIdx.y<N)
        {
            tileB[threadIdx.y][threadIdx.x] = B[(tile*TILE_WIDTH + threadIdx.y)*N + col];
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        float sum = 0.0f;
        for(int k=0;k<TILE_WIDTH;k++)
        {
            sum += tileA[threadIdx.y][k]*tileB[k][threadIdx.x];
        }

        __syncthreads();

        if(row<N && col<N)
        {
            C[row*N + col] += sum;
        }
    }

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

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
  

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
