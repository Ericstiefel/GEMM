#include "gemm.h"
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <chrono> 
#include <cuda_runtime.h>
#include <cmath> 

// --- Configuration ---
const int M = 1024; // Rows of A and C
const int K = 1024; // Cols of A and Rows of B
const int N = 1024; // Cols of B and C
const int TILE_WIDTH = 32; // Tile width for the tiled GPU implementation

// --- CUDA Error Checking Macro ---
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// --- Helper function to initialize matrices ---
void initializeMatrix(float* mat, int rows, int cols) {
    std::mt19937 mt(static_cast<unsigned int>(time(0)));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dist(mt);
    }
}

// --- Helper function to verify results ---
void verify_result(const float* cpu_res, const float* gpu_res, int m, int n) {
    const float tolerance = 1e-5;
    for (int i = 0; i < m * n; ++i) {
        if (fabs(cpu_res[i] - gpu_res[i]) > tolerance) {
            std::cerr << "Verification FAILED at index " << i
                      << "! CPU: " << cpu_res[i]
                      << ", GPU: " << gpu_res[i] << std::endl;
            return;
        }
    }
    std::cout << "Verification PASSED!" << std::endl;
}

int main() {
    // --- Host Memory Allocation ---
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu_naive, *h_C_gpu_tiled;
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C_cpu = new float[M * N];
    h_C_gpu_naive = new float[M * N];
    h_C_gpu_tiled = new float[M * N];

    // --- Initialize Host Matrices ---
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);
    std::cout << "Matrices (" << M << "x" << K << " and "
              << K << "x" << N << ") generated successfully." << std::endl;

    // --- Device Memory Allocation ---
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // --- Copy Data from Host to Device ---
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // --- Profiling Events ---
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float gpu_time = 0.0f;

    // =======================================================
    // 1. Profile CPU Implementation
    // =======================================================
    std::cout << "\nRunning CPU implementation..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gemm(h_A, h_B, h_C_cpu, M, K, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;

    // =======================================================
    // 2. Profile Naive GPU Implementation (BASELINE)
    // =======================================================
    std::cout << "\nRunning Naive GPU implementation (Baseline)..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    naive_launcher(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    std::cout << "Naive GPU Time: " << gpu_time << " ms" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C_gpu_naive, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    verify_result(h_C_cpu, h_C_gpu_naive, M, N);

    // =======================================================
    // 3. Profile Tiled GPU Implementation (COMPARISON)
    // =======================================================
    std::cout << "\nRunning Tiled (Double Buffered) GPU implementation (Comparison)..." << std::endl;
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float))); // Clear device memory for C
    CUDA_CHECK(cudaEventRecord(start));
    tiled_buff_launcher<TILE_WIDTH>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    std::cout << "Tiled GPU Time: " << gpu_time << " ms" << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C_gpu_tiled, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    verify_result(h_C_cpu, h_C_gpu_tiled, M, N);

    // --- Cleanup ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu_naive;
    delete[] h_C_gpu_tiled;

    return 0;
}
