#include <stdio.h>
#include <cuda_runtime.h>
#define N 1000


__global__ void matrixAdd(
    float *A,
    float *B,
    float *C)
{

    int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;


    int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;


    int index =
        row * N + col;


    if(row < N && col < N)
    {
        C[index] =
            A[index] + B[index];
    }
}



int main()
{

    size_t size =
        N*N*sizeof(float);


    float *A;
    float *B;
    float *C;


    // CPU memory

    A=(float*)malloc(size);
    B=(float*)malloc(size);
    C=(float*)malloc(size);



    for(int i=0;i<N*N;i++)
    {
        A[i]=1.0;
        B[i]=2.0;
    }



    float *d_A;
    float *d_B;
    float *d_C;



    // GPU memory

    cudaMalloc(
        &d_A,
        size);

    cudaMalloc(
        &d_B,
        size);

    cudaMalloc(
        &d_C,
        size);



    // CPU -> GPU

    cudaMemcpy(
        d_A,
        A,
        size,
        cudaMemcpyHostToDevice);


    cudaMemcpy(
        d_B,
        B,
        size,
        cudaMemcpyHostToDevice);



    // define block

    dim3 block(
        16,
        16);


    dim3 grid(
        (N+15)/16,
        (N+15)/16);



    matrixAdd<<<grid,block>>>(
        d_A,
        d_B,
        d_C);



    cudaDeviceSynchronize();



    // GPU -> CPU

    cudaMemcpy(
        C,
        d_C,
        size,
        cudaMemcpyDeviceToHost);



    printf(
        "C[0]=%f\n",
        C[0]);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    free(A);
    free(B);
    free(C);


    return 0;
}