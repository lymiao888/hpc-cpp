#include <cstdio>

__global__ void kernel() {
    printf("Hello, world!\n");
}

int main() {
    kernel<<<1, 1>>>();
    return 0;
}
//没有反应，因为CPU和GPU是异步的
//需要加cudaDeviceSynchronize();