#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <vector>
// #include <thrust/device_vector.h>  // 如果想用 thrust 也是没问题的

// 这是基于“边角料法”的，请把他改成基于“网格跨步循环”的：10 分
__global__ void fill_sin(int *arr, int n) {
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        arr[i] = sinf(i);
    }
}

__global__ void filter_positive(int *counter, int *res, int const *arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) return;
    if (arr[i] >= 0) {
        // 这里有什么问题？请改正：10 分
        // int loc = *counter;
        // *counter += 1;
        int loc = atomicAdd(counter, 1); // 改成了原子操作
        res[loc] = n;
    }
}

int main() {
    constexpr int n = 1<<24;
    std::vector<int, CudaAllocator<int>> arr(n);
    std::vector<int, CudaAllocator<int>> res(n);
    int *pcounter;
    int counter;
    checkCudaErrors(cudaMalloc(&pcounter, sizeof(int)));

    // fill_sin 改成“网格跨步循环”以后，这里三重尖括号里的参数如何调整？10 分
    int nthreads = 128;
    int nblocks = (n + nthreads + 1) / nthreads;
    fill_sin<<<nblocks, nthreads>>>(arr.data(), n);

    // 这里的“边角料法”对于不是 1024 整数倍的 n 会出错，为什么？请修复：10 分
    filter_positive<<<n / 1024, 1024>>>(pcounter, res.data(), arr.data(), n);

    // 这里 CPU 访问数据前漏了一步什么操作？请补上：10 分
    checkCudaErrors(cudaMemcpy(&counter, pcounter, sizeof(int), cudaMemcpyDeviceToHost));
    if (counter <= n / 50) {
        printf("Result too short! %d <= %d\n", counter, n / 50);
        return -1;
    }
    for (int i = 0; i < counter; i++) {
        if (res[i] < 0) {
            printf("Wrong At %d: %f < 0\n", i, res[i]);
            return -1; 
        }
    }

    printf("All Correct!\n"); 
    cudaFree(pcounter);
    return 0;
}