# 从汇编角度看编译器优化
## 0 汇编语言

32位时代x86通用寄存器：eax, ecx, edx, ebx, esi, edi, esp, ebp
64位时代x86通用寄存器：rax, rcx, rdx, rbx, rsi, rdi, rsp, rbp, r8, r9, r10, r11, ..., r15

常见的汇编指令和一些小知识 :
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
10. SIMD原理 : single-instruction multiple-data
    SIMD把4个float打包到一个xmm寄存器里面，类似矢量运算；一定条件下编译器可以把一个处理标量float的代码转换成一个利用SIMD处理矢量float的代码 

## 1 化简

编译器优化
1. 代数化简 : 简化代数表达式
2. 常量折叠 : 对常量直接计算然后放到寄存器
3. 避免代码复杂化 : 避免会造成new/delete的容器--即内存分配在堆上的容器
储存在堆上 : vertor,map,set,string,function,any,unique_ptr,shared_ptr,weak_ptr,...
储存在栈上 : array,bitset,glm:vec,string_view,pair,tuple,optional,variant,...
Q 储存在堆和栈上有什么区别？


1. 
## 2 内联
## 3 指针
## 4 矢量化
## 5 循环
## 6 结构体
## 7 STL容器
