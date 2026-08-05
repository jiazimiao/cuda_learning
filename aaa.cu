#include <stdio.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <cuda/cmath>
#include <cuda_runtime.h>
#define N 4*1024

__global__ void matrixAdd(
    float *A,
    float *B,
    float *C)
{

    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    int index =
        row * N + col;

    if (row < N && col < N)
    {
        C[index] =
            A[index] + B[index];
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

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
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

    // dim3 block(16, 16);
    // dim3 grid((N + 15) / 16, (N + 15) / 16);
    dim3 block(1024,1024);
    dim3 grid(4,4);

    matrixAdd<<<block, grid>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // GPU -> CPU

    cudaMemcpy(
        C,
        d_C,
        size,
        cudaMemcpyDeviceToHost);

    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}