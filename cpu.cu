#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

// CPU进行矩阵计算
void matrixMulCpu(float* fpMatrixA, float* fpMatrixB, float* fpMatrixC, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += fpMatrixA[i * k + l] * fpMatrixB[l * n + j];
            }
            fpMatrixC[i * n + j] = sum;
        }
    }
}

int main() {
    // 设置伪随机数生成器种子
    srand(static_cast<unsigned>(time(nullptr)));

    // 设置矩阵维度
    int m = 100;  // 矩阵A的行数
    int n = 100;  // 矩阵B的列数
    int k = 100;  // 共享维度

    // 分配内存并生成随机的稠密矩阵A和B
    float* matrixA = new float[m * k];
    float* matrixB = new float[k * n];
    float* matrixC = new float[m * n];

    for (int i = 0; i < m * k; i++) {
        matrixA[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < k * n; i++) {
        matrixB[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 执行矩阵乘法计算，并测量执行时间
    auto start_time = std::chrono::high_resolution_clock::now();

    matrixMulCpu(matrixA, matrixB, matrixC, m, n, k);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // 输出执行时间
    std::cout << "Matrix multiplication took " << duration.count() << " milliseconds." << std::endl;

    // 释放内存
    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;

    return 0;
}
