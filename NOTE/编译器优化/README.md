# 从汇编角度看编译器优化
## 0 汇编语言

32位时代x86通用寄存器：eax, ecx, edx, ebx, esi, edi, esp, ebp
64位时代x86通用寄存器：rax, rcx, rdx, rbx, rsi, rdi, rsp, rbp, r8, r9, r10, r11, ..., r15

### 常见的汇编指令和一些小知识 :
1. eax : 返回值通过eax传出
2. movl %edi, %eax : 相当于eax = edi
3. imull : 32位乘法，imull %esi, %eax 相当于 eax *= esi
4. imulq : 64位乘法，同上，不过是int64_t
5. leal : 加载表达式地址，相当于 & 
   看这个, (%rdi,%rsi) 相当于 *(rdi + rsi)
   那么 leal (%rdi,%rsi), %eax 相当于 ？
   eax = &*(rdi + rsi) 
   整数加法被优化成 leal
6. movslq 把32位符号扩展到64位
7. size_t : 在64位系统上相当于unit_64t，在32位系统上相当于unit_32t，这样就可以不用movslq
8. xmm系列寄存器 : 存浮点，128位宽，可以容4个float，或2个double
9. addss : 第一个s表示标量（可以用p表示矢量），第二个s表示单精度浮点数（可以用d表示双精度浮点数）
    省流：如果汇编大量ss说明矢量化失败，大多数ps说明矢量化成功
10. call: 调用外部函数 eg. _Z5otheri@PLT 这里@PLT是函数链接表，编译器会查找其他的.o文件是否定义_Z5otheri，如果有就就把@PLT替换成他的地址
如果开编译器优化会怎么样？call会变成jmp
* jmp可以直接段内近转移
* call相当于将IP压入栈中，再jmp；一般会配合ret，ret弹出栈里面的内存赋给IP
需要知道 : cs是代码段寄存器，ds是数据段寄存器。cs * 16 + IP 得到代码段地址； ds * 16 得到数据段地址
这就是为什么段内跳转只需要改IP
11. movups : 代表寄存器地址不一定对齐到16字节
    movaps : 代表寄存器地址对齐到16字节
12. memset : 数组清零
### SIMD原理 : single-instruction multiple-data
SIMD把4个float打包到一个xmm寄存器里面，类似矢量运算；一定条件下编译器可以把一个处理标量float的代码转换成一个利用SIMD处理矢量float的代码 

## 1 化简

### 编译器优化
1. 代数化简 : 简化代数表达式
2. 常量折叠 : 对常量直接计算然后放到寄存器
3. 避免代码复杂化 : 避免会造成new/delete的容器--即内存分配在堆上的容器
储存在堆上 : vertor,map,set,string,function,any,unique_ptr,shared_ptr,weak_ptr,...
储存在栈上 : array,bitset,glm:vec,string_view,pair,tuple,optional,variant,...
Q 储存在堆和栈上有什么区别？储存在栈上无法扩充大小
注 : 如果语句过多，编译器可能会放弃优化
4. constexpr : 强迫编译器在编译期求值
注 : constexpr函数中不能用非constexpr容器（储存在堆上的那些）
5. 合并写入 : 可以把两个int32的写入合并为一个int64的写入（前提地址没有跳跃）
## 2 内联
* 什么是内联？当编译器看得到被调用函数的时候，会直接把函数贴到调用他的函数里面
* ⚠️！只有定义在同一个文件的函数可以被内联
* 用static，inline无所谓的 why？
  1. static是为了避免多个.cpp文件引用同一个头文件造成冲突，并不是必须
  2. 内联与否跟inline没关系，只取决于是否在同文件，而且函数体够小

## 3 指针
* 指针别名现象 pointer aliasing
__restrict 关键字 : 向编译器保证这些指针不会发生重叠
注意：__restrict 只用加在非const的前面就可以，const本身就是禁止写入的
* volatile 关键字 禁止优化 （有时候用来做性能测试）
* 非常重要的注意 : __restrict 和 volatile 的区别
  语法上：volatile 在*前面，
        __restrict 在*后面
  volatile和const都是C++标准的一部分，但是__restrict是c99标准关键字

## 4 矢量化
### SIMD来了
SIMD是一个典型的合并写入
* 能多宽？两个int32可合并为一个int64；四个int32可合并为一个__m128；八个int32可合并为一个__m256
* ymm系列寄存器：AVX向量 （可以在gcc编译命令中加"-march=native"让编译器自动检测当前硬件支持的指令集）
* 边界如果不是4的倍数要做特判

### 相关汇编语言
* paddd 四个int的加分
* movdqa 加载四个int
## 5 循环loops

* OpenMP强制矢量化
```C++
#pragma omp simd
    for (int i = 0; i < 1024; i++) {
        a[i] = b[i] +1;
    }
```
* 还可以用下面的代码来表示忽视下方for循环中使用的可能的指针别名，不同编译器不同
```
#pragma GCC ivdep
```
把循环中的不变量挪到外面来

* 循环展开
```C++
#pragma GCC unroll 4
```
### 什么时候会优化失败？
1. 为什么循环中 b[i] * dt * dt 优化失败了？因为乘法是做结合的 (b[i] * dt) * dt 识别不到不变量
2. 调用不在另一个文件的函数
3. 循环中下标跳跃或者随机

## 6 结构体
如果结构体没有内存对齐的话也会出现优化失败的情况
解决办法 : 在结构体struct后加上alignas(要对齐的字节数)
### 结构体的内存布局
#### AOS 
当个对象的属性紧挨着存放
#### SOA
属性分离的储存在多个数组里面
这就是为什么AOS需要对齐到2的幂才能优化矢量化

SOA + unroll?

## 7 STL容器
发现std::vector也会有指针别名的问题
解决办法 : #pragma omp simd 或 #pragma GCC ivdep

## 8 数学运算
1. 除法变成乘法
2. 分离公共除法 可以用手动优化，或者加编译选项 -ffast-math
3. 数学函数加std::前缀
std::sqrt重载了float和double，而sqrt只接受double
std::abs重载了int，float和double

### 嵌套循环
直接累加，会有指针别名的问题，可以尝试先读到局部变量，累加完成之后再写入