#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// CUDA错误检查宏
#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(-1); \
    }
    
__global__ void matrixMulSharedKernel(float* fpMatrixA, float* fpMatrixB,float* fpMatrixC, int m, int n, int k)
{
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    int BLOCK_SIZE=32;
    __shared__ float shTileA[32][32];
    __shared__ float shTileB[32][32];

    int nIter = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for(int i = 0; i < nIter; i++)
    {
        // load data from global memory to shared memory
        shTileA[threadIdx.y][threadIdx.x] = fpMatrixA[nRow * k + i * BLOCK_SIZE + threadIdx.x];
        shTileB[threadIdx.y][threadIdx.x] = fpMatrixB[(i * BLOCK_SIZE + threadIdx.y) * n + nCol];

        // sync to wait for all threads in one block to finish loading datas
        __syncthreads();

        // sub-matrix multiply
        for(int l = 0; l < BLOCK_SIZE; l++)
        {
            fCVal += shTileA[threadIdx.y][l] * shTileB[l][threadIdx.x];
        }

        // sync to wait for all threads in one block to finish compute
        __syncthreads();
    }

    // store results into global memory
    fpMatrixC[nRow * n + nCol] = fCVal;
}
int main() {
    // 设置矩阵维度和分配内存
    int m =2000;       //矩阵A的行数
    int n = 2000;       //矩阵B的列数
    int k = 2000; // 共享维度
    int sizeA = m * k * sizeof(float);
    int sizeB = k * n * sizeof(float);
    int sizeC = m * n * sizeof(float);

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // 初始化主机端矩阵
    float* hostMatrixA = new float[sizeA];
    float* hostMatrixB = new float[sizeB];
    float* hostMatrixC = new float[sizeC];

    for (int i = 0; i < sizeA; i++) {
        hostMatrixA[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < sizeB; i++) {
        hostMatrixB[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 将矩阵数据从主机复制到设备
    cudaMemcpy(d_A, hostMatrixA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, hostMatrixB, sizeB, cudaMemcpyHostToDevice);

    // 配置线程块和网格
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));

    // 记录开始时间
    CUDA_CHECK_ERROR(cudaEventRecord(start));

    // 执行GPU核函数
    matrixMulSharedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);

    // 记录结束时间
    CUDA_CHECK_ERROR(cudaEventRecord(stop));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop));

    // 将结果从设备复制回主机
    cudaMemcpy(hostMatrixC, d_C, sizeC, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 计算执行时间
    float milliseconds = 0;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "GPU Execution Time: " << milliseconds << " ms" << std::endl;

    // 释放主机内存
    delete[] hostMatrixA;
    delete[] hostMatrixB;
    delete[] hostMatrixC;

    return 0;
}