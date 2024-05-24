# Cuda编程
官方文档指路:[NVIDIA CUDA 编程指南](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf)

## 01 简介
### Q&A
**1. Cmake怎么支持CUDA？**
CMake版本在3.18以上的话，在LANGUAGES后面加CUDA就可以了  ```project(main LANGUAGES CXX CUDA)```

**2. 为什么下面的代码运行不出来print？**
```C++
#include <cstdio>
__global__ void kernel() {
    printf("Hello, world!\n");
}
int main() {
    kernel<<<1, 1>>>();
    return 0;
}
```
用```__global__```修饰的就是核函数，没有打印是因为CPU和GPU之间的通信是异步的。也就是说，CPU调用```kernal<<<1, 1>>>()```并不会在GPU上执行完了之后返回，实际上是把kernal这个任务推送给GPU的执行队列上，然后立即返回
**解决** cudaDeviceSynchronize()等待GPU完成队列的所有任务之后再返回

**3. 怎么指定在GPU上运行？**
__\_\_global\_\___ 用于定义核函数，它在GPU上执行，从CPU端调用，需要<<<>>>，可以有参数，不能有返回值
__\_\_device\_\___ 用于定义设备函数，它在GPU上执行，但是从GPU上调用，不需要<<<>>>,可以有参数，不能有返回值
__\_\_host\_\___ 用于定义主机函数，它在CPU上执行，任何函数如果没有指明修饰符，默认host

**4. 怎么同时在CPU和GPU上运行？**
* 通过双重修饰符 __\_\_host\_\___ __\_\_device\_\___
* 让constexpr函数自动变成CPU和GPU都可以调用的函数，
> constexpr函数啥玩意？
> 1. 表达式可以在编译时被确定
> 2. 蕴含了顶层const，即值固定
> 3. 如果要用constexpr+cuda，我们的CMakelist.txt里面要加```target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)```
> 值得注意的是constexpr函数通常是一些可以内联的函数，不能调用printf这样的函数，也不能调用__syncthreads之类的GPU特有函数，因此不能完全代替 \_\_host\_\_ \_\_device\_\_

**5. 怎么让一个函数针对GPU和CPU有不用的指令？**
```C++
__host__ __device__ void say_hello(){
#ifdef __CUDA_ARCH__
    printf("hello, world from GPU!\n");
#else
    printf("hello, world from GPU!\n");
#endif
}
```
原理：CUDA编译器具有多段编译的特点，一段代码他会先送到CPU上编译生成CPU部分指令码，然后送到真正的GPU编译器生成GPU指令码。最后链接生成同一个文件看起来只编译一次，实际上预处理了很多次，在GPU编译模式下定义_-CUDA_ARCH__这个宏，利用#ifdef判断这个宏有没有定义

**6. 啥是__CUDA_ARCH__?**
它是一个整数，表示当前编译所针对的GPU的架构版本号是多少，520就是5.2.0
可以通过CMake设置架构版本号```set(CMAKE_CUDA_ARCHITECTURES 75)```




