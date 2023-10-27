#include <iostream>
#include "cutlass/include/cutlass/gemm/device/gemm.h"
#include <cstdlib>
#include <chrono>

using namespace std;
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

using CutlassGemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

void generate_tensor_2D(float *ptr, int M, int N) {
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            *(ptr + i * N + j) = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

int main(int argc, const char *arg[]) {
    int M = 1000;
    int N = 1000;
    int K = 1000;

    int lda = K;
    int ldb = K;
    int ldc = N;
    int ldd = N;

    float alpha = 1.0;
    float beta = 1.0;

    float *A;
    float *B;
    float *C;
    float *D;

    size_t A_mem_size = sizeof(float) * M * K;
    size_t B_mem_size = sizeof(float) * K * N;
    size_t C_mem_size = sizeof(float) * M * N;
    size_t D_mem_size = sizeof(float) * M * N;

    A = (float*)malloc(A_mem_size);
    B = (float*)malloc(B_mem_size);
    C = (float*)malloc(C_mem_size);
    D = (float*)malloc(D_mem_size);

    generate_tensor_2D(A, M, K);
    generate_tensor_2D(B, K, N);
    generate_tensor_2D(C, M, N);

    float *d_A;
    float *d_B;
    float *d_C;
    float *d_D;

    cudaMalloc((void**)&d_A, A_mem_size);
    cudaMalloc((void**)&d_B, B_mem_size);
    cudaMalloc((void**)&d_C, C_mem_size);
    cudaMalloc((void**)&d_D, D_mem_size);

    cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, C_mem_size, cudaMemcpyHostToDevice);

    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({M, N, K}, {d_A, lda}, {d_B, ldb}, {d_C, ldc}, {d_D, ldd}, {alpha, beta});

    auto start = chrono::high_resolution_clock::now();

    gemm_operator(args);

    auto stop = chrono::high_resolution_clock::now();

    cudaMemcpy(D, d_D, D_mem_size, cudaMemcpyDeviceToHost);

    chrono::duration<double> duration = stop - start;
    cout << "GEMM Execution Time: " << duration.count() << " seconds" << endl;

    free(A);
    free(B);
    free(C);
    free(D);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
}
