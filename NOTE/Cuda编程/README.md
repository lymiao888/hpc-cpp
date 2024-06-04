# Cuda编程
官方文档指路:[NVIDIA CUDA 编程指南](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf)

## 00 简介
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

## 01 线程与板块
### 线程
**<<<板块数量，每个板块中的线程数量>>>**  这就是三重尖括号里的数字的含义呀
* 获取线程编号：**threadIdx.x** 这个是CUDA的特殊变量之一，只有在核函数里面才能被访问
* 获取线程数量：**blockDim.x**
### 板块
一个板块可以有多个线程组成
* 获取板块编号：**blockIdx.x**
* 获取板块数量：**gridDim.x**

动用数学算算展平？
总的线程数量：blockDim * gridDim
总的线程编号：blockDim * blockIdx + threadIdx

三维的板块的线程编号 dim3(x, y, z)
二维的板块的线程编号 dim3(x, y, 1)
一维的板块的线程编号 dim3(x, 1, 1) 

**分离 __device__ 函数的声明和定义解决报错**
tips1：开启```set(CMAKE_CUDA_SEPARABLE_COMPILATION)```

## 02 内存管理

### 内存问题的出现
* 正如01的第三个问题所描述的，kernel的调用是异步的，返回的时候不会实际让GPU把核函数执行完，而是用```cudaDeviceSynchronize()```等待它执行完毕，所以**不能从kernel里通过返回值传给GPU**，所以核函数的返回类型必须是void
* 怎么解决？？ 试图用指针传递，并不可靠，错误代码怎么分析？--用helper_cuda.h里的checkCudaErrors()
```C++
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
__global__ void kernal(int *pret){
    *pret = 42
}
int main() {
    int ret = 0;
    kernel<<<1, 1>>>(&ret);
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
```
> 报错：
> CUDA error at /home/lym/hpc-cpp/NOTE/Cuda编程/cuda-learning/02_memory/04/main.cu:12 code=700(cudaErrorIllegalAddress) "cudaDeviceSynchronize()" 
思考是否是因为ret建立在栈上，所以GPU不能访问？
```C++
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
__global__ void kernal(int *pret){
    *pret = 42
}
int main() {
    int *pret = (int *)malloc(sizeof(int));
    kernel<<<1, 1>>>(pret);
    checkCudaErrors(cudaDeviceSynchronize());
    free(pret);
    return 0;
}
```
* 还是一样的报错QAQ怎么办？
* 答案：**GPU使用的是独立的显存，不能访问CPU内存**，CPU内存叫主机内存host，GPU内存叫设备内存device，所以无论是栈还是malloc分配的都是CPU上的内存，无法被GPU访问到
  可以用```cudaMalloc()```分配GPU上的显存，结束的时候```cudaFree()```
  同理，**CPU也不能访问GPU的内存地址**

### 跨GPU/CPU地址拷贝数据解决办法
1. 用cudaMalloc来分配GPU上的显存
2. 用cudaMemcpy在GPU和CPU之间拷贝数据
```C++
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
__global__ void kernel(int *pret){
    *pret = 42
}
int main(){
    int *pret;
    checkCudaErrors(cudaMalloc(&pret, sizeof(int)));
    kernel<<<1, 1>>>(pret);
    checkCudaErrors(cudaDeviceSynchronize());

    int ret;
    checkCudaErrors(cudaMemcpy(&ret, pret, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(pret);
    return 0;
}
```
同理，还有```cudaMemcpyDeviceToDevice```, ```cudaMemcpyHostToDevice```
```cudaMemcpy```会自动同步，和cudaDeviceSynchronize()等价

3. 统一内存地址技术（Unified Memory）
只需要把cudaMalloc换成cudaMallocManaged就可以咯，这样有什么好处或者说区别呢？
这样分配出来的地址不论在CPU上还是GPU上都是一模一样的，都可以访问，并且当从CPU访问时拷贝也会按需进行不需要手动调用cudaMemcpy
   
### 总结
| 内存类型  |       开辟内存    | 释放内存    |
|:-------:|:-----------------:|:---------:|
| host    | malloc            | free    |
| device  | cudaMalloc        | cudaFree|
| managed | cudaMallocManaged | cudaFree|

## 03 数组
* 怎么分配数组的空间呢？
可以用 cudaMalloc 配合 n * sizeof(int) ，比如 ```cudaMallocManaged(&arr, n * sizeod(int))``` 
<!-- 感觉学不下去了，这个cuda的我学了好久又感觉啥也没学，啊啊啊啊啊我该怎么办，先睡20分钟吧 -->

tips：
* 网格跨步循环（grid-stride loop）
```C++
__global__ void kernel(int *arr, int n) {
    for(int i = threadIdx.x; i < n; i += blockDim.x) {
        arr[i] = i;
    }
}
```
啥意思，就是说无论指定了多少个线程，都能在n区间里面循环，避免越界+漏元素
* 从线程到板块
```C++
int nthreads = 128;
int nblocks = n / nthreads;
kernel<<<nblocks, nthreads>>>(arr, n);
```
* 边角料问题--向上取整的除法
```C++
int nthreads = 128;
int nblocks = (n + nthreads + 1) / nthreads;
kernel<<<nblocks, nthreads>>>(arr, n);
```

## 04 Cuda与C++封装性
```std::vector```作为模版类，其实有两个模版参数，```std::vector<T,Allocator T>```
我们平时用的就是```std::vector<T>```，因为第二个参数默认是```std::allocator<T>```，等价于```std::vector<T,std::allocator<T>>```
```std::allocator<T>```负责分配和释放内存
他有几个功能函数```T *allocate(size_t n)```,```void deallocate(T *p, size_t n)```

> 好复杂啊，简单来说，就是vector这个容器依赖于Allocator来分配和释放内存，默认是CPU，但是我们要在GPU上使用vector就要改再改ALlocator
// (源代码代码来自@archibate)他写一个了cudaAllocator骗过vector；当然我们最好是用CUDA官方给的thrust库比较好
```C++
#include <cstddef>
#include <utility>
#include <cuda_runtime.h>
#include "helper_cuda.h"

template <class T>
struct CudaAllocator {
    using value_type = T;

    T *allocate(size_t size) {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        checkCudaErrors(cudaFree(ptr));
    }
    template <class ...Args>
    void construct(T *p, Args &&...args) {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))
            ::new((void *)p) T(std::forward<Args>(args)...);
    }

    constexpr bool operator==(CudaAllocator<T> const &other) const {
        return this == &other;
    }
};
```
解释一下后面两个函数：目的是禁用0初始化，如果无参，则跳过，从而避免在CPU上的低效0初始化
咋用？eg. ```std::vector<int, CUdaAllocator<int>> arr(n);```

### template
```C++
template <int N, class T>
__global__ void kernel(T *arr) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}
```
怎么调用？```kernel<n><<<32, 128>>>(arr.data())```注意如果手动指定模版参数，请放在三重尖括号前面

### functor
核函数可以实现函数式编程
```C++
template <class Func>
__global__ void parallel_for (int n, Func func){
    for (int i = blockDim.x * blockIdx.x + threadidx.x; i < n; i += blockDim.x * gridDim.x)
        func(i);
}
struct Functor { 
    __device__ void operator()(int i) const {
        printf("number %d", i);
    }
}
int main() {
    int n = 65555;
    parallel_for<<<32, 128>>>(n, Functor{});
    checkCudaErrors(cudaDeviceSynchronize());
    return 0;
}
```
注意:
1. Func不能是Fun const&，不然会编程一个指向CPU地址的指针，CPU向GPU传参数必须要按值传递
2. 作为参数的函数必须是一个有成员函数operator()的类型，不能是独立的函数
3. 这个函数必须被标记为__device__类型，不然会在cpu上
### lambda
这个很复杂，首先我们需要在编译环境中打开 --extended_lambda
在CMake中加入```target_compile_options(mian PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)```
1. 基础版
```C++
parallel_for<<<32, 128>>>(n, [] __device__ (int i) {
    printf("number %d", i);
});
```
2. 加强版--获取外部变量
parallel_for函数的定义还是如上
* 我们知道lambda函数可以用[&]捕获变量，但是如果直接这样捕获会出错，因为此时捕获的是CPU堆栈上的变量本身，而我们的核函数需要arr在GPU上的内存地址
* 我们可以用[=]按值捕获来解决吗？答案是不行，因为vector的拷贝是深拷贝，这样会把整个vector都拷贝到GPU上，显然这不是我们的目的，我们的目的是在核函数中把GPU上的arr进行赋值，而不改变CPU上的arr
**解决方案1**
```C++
int main() {
    int n = 65536;
    std::vector<int, CUdaAllocator<int>> arr(n);
    int *arr_data = arr.data();
    parallel_for<<<32, 128>>>(n, [=] __device__ (int i)) {
        arr_data[i] = i;
    }
    checkCudaErrors(cudaDeviceSynchronize());
}
```
data()返回一个起始地址的原始指针，这个指针是浅拷贝的，所以可以拷贝到GPU上让他访问
> 啥？浅拷贝就行，深拷贝就不行？这是啥意思
> 重新总结一下这个点，首先需要回顾之前说cuda内存分配的知识，GPU不能访问CPU的内存，同样CPU也不能访问GPU的内存，我们核函数传入的lambda函数要对GPU上的数据进行操作的时候，就要注意```std::vector<int, CUdaAllocator<int>> arr(n);```这个的含义是我们在CPU上定义了arr这个变量，但是他的数据在GPU上，所以我们需要取数据的话，需要对arr浅拷贝，获得指针


**解决方案2**
```C++
int main() {
    int n = 65536;
    std::vector<int, CUdaAllocator<int>> arr(n);
    parallel_for<<<32, 128>>>(n, [arr = arr.data()] __device__ (int i)) {
        arr[i] = i;
    }
    checkCudaErrors(cudaDeviceSynchronize());
}
```
这个跟上面的是一样的，只是我们可以在核函数内直接用arr变量名

### 总结
C++的重要特性就体现在STL，模版，lambda函数这几个方面，上面的例子展现了C++的封装性，cuda对C++的支持使得这个几个C++的特点都可以在核函数上实现，这个是非常重要的
## 05 thrust库
1. thrust分配内存
先来看看一个小程序
```C++
#include <turust/universal_vector.h>
int main(){
    int n = 65536;
    thrust::universal_vector<float> x(n); // 统一内存非配，无论CPU还是GPU都能访问
    thrust::host_vector<float> y(n);// host_vector帮我们在cpu上分配内存
    thrust::device_vector<float> z(n);// device_vector帮我们在gpu上分配内存
    // 可以通过 = 运算符实现上面两者的数据拷贝，可以自动调用cudaMemcpy
}
```
2. thrust提供模版函数
下面的几个函数是对迭代器的操作
* thrust::generate(m, n, func)
* thrust::for_each(m, n, func)

## 06 原子操作
还记得之前C++中原子操作吗，因为多线程并行的时候```sum += arr[i]``` 这样的操作其实分成了四步，
1. 读取sum到寄存器A
2. 读取arr[i]到寄存器B
3. 让寄存器A的值加上寄存器B的值
4. 把寄存器A的值写回到sum内存中
所以如果并发操作会出现大问题，这也是C++原子操作诞生的目的，通过让这几个操作不可分割，保证计算的准确性

Cuda的核函数也有原子操作
atomicAdd() atomicSub() ... ...

> ```atomicCAS```我们可以通过这个来构造新的原子操作
> 比如```atomicCAS(a, b, a * b)```


## 08 板块与共享内存

## 08 共享内存进阶

## 09 插桩
