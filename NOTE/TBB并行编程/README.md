# TBB并行编程

## 0 从并发到并行
### 介绍
性能不是线性增长的哦，为什么？因为为了保存缓存一致性以及其他握手协议需要运行的开销。
* 并发: 单核处理器，操作系统通过时间调度，轮换着进行不同的线程，看起来好像是同时运行，其实每个时刻只有一个线程在运行，目的是异步处理多个不同的任务，避免同步造成阻塞
* 并行: 多核处理器，每个处理器执行一个线程，真正的同时进行
### TBB的并行例子
同时“下载”和“用户交互”按之前学的可以用thread开两个线程，等待下载join
1. task_group
```C++
#include <tbb/task_group.h>
#include <string>
#include <iostream>
int main(){
    tbb::task_group tg;
    tg.run([&] {
        download("hello.zip");
    });
    tg.run([&] {
        interact();
    });
    tg.wait();
    return 0;
}
```
基于TBB版本，区别是一个任务不一定对应一个线程，如果任务数超过CPU最大的线程数，TBB会在用户层负责调度任务
2. parallel_invoke 封装好了捏
```c++
#include <tbb/parallel_invoke.h>
#include <string>
#include <iostream>
int main(){
    tbb::parallel_invoke([&]{download("hello.zip");},[&]{interact();});
    return 0;
}
```
## 1 并行循环

1. parallel_for
2. parallel_for_each
3. blocked_range2d/blocked_range3d

## 2 缩并与扫描

1. parallel_reduce
2. parallel_deterministic_reduce
3. paralle_scan

## 3 性能测试

## 4 任务阈与嵌套

## 5 任务分配

## 6 并发容器

## 7 并行筛选

## 8 流水线并行
