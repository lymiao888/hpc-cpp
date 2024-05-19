# c++11开始的多线程编程

## 0 时间
c++ 引入标准库 ```std::chrono```
1.  时间点VS时间段
* 时间点 : ```chrono::steady_clock::time_point```
* 时间段 : ```chrono::milliseconds```, ```chrono::seconds```, ```chrono::minutes```
2. 对时间段的处理 
```C++
std::chrono::duration_cast<T,R>
```
```C++
using double_ms = std::chrono::duration<double, std::milli>;
double ms = std::chrono::duration_cast<double_ms>(dt).count();
```
3. 开睡！sleep
用```std::this_thread::sleep_for```代替Unix类专用的usleep，如下面例子
```C++
std::this_thread::sleep_for(std::chrono::milliseconds(400));
```
如果要睡到时间点，如下例子
```C++
std::this_thread::sleep_until(t);
```
## 1 线程
### 概念
1. 进程和线程的概念
* 进程 : 一个程序被操作系统拉起来加载到内存之后从开始执行到执行结束的一个过程 
* 线程 : 是进程的一个实体，是被系统独立分配和调度的基本单位
注意：线程是CUP可以执行调度的最小单元，进程本身不能获得CPU时间，但是线程可以
2. 进程和线程的区别和联系
* 从属关系 : 一个进程可以有多个线程
* 每一个线程共享同样的内存空间，所以开销小
* 每一个进程拥有独立的内存空间，因此开销大
3. 为什么需要多线程？
因为我们需要无阻塞多任务

### 怎么用多线程（此处建议看代码）
1. std::thread类表示线程，构造函数参数可以是lambda表达式
注意 : std::thread 是基于 pthread，所以linking的时候我们要链接Threads::Threads
我们的CmakeLists.txt可以加上如下内容
```CMake
find_package(Threads REQUIRED)
target_link_libraries(test PUBLIC Threads::Theards)
```
2. 主线程等待子线程结束，t1.join()
3. std::thread的解构函数会销毁线程
作为一个C++类，它RAII思想，删除了拷贝构造/赋值函数，但提供了移动构造/赋值函数
4. 让解构函数不再销毁线程，t1.detach()
   * 用detach()成员函数分离线程的含义指的是，线程的生命周期不再有当前std::thread对象管理，而是在线程退出后自动销毁自己
   * 但是detach的问题是进程退出的时候不会等待所有子线程执行完毕
5. 让解构函数不再销毁线程，移动到全局线程池
   * 这样做的原因是，延长线程的额生命周期到函数外部
   * 可以自己定一个ThreadPool类，在解构的时候join
6. c++20 引入```std::jthread```类，在解构的时候自动join

## 2 异步
1. ```std::async``` 接受一个带返回值的lambda函数，返回一个```std::future```对象（就是现在没有值但是之后肯定会有，然后lambda绘制另一个线程里，之后调用future的get()
   * future还可以调用wait(),wait_for(),wait_until()

2. ```std::async``` 的第一个参数设置为 ```std::launch:deferred```, 这时不会创建一个线程，只会把lambda函数的运算推迟到future的get(), 这只是函数式编程范式的异步
3. ```std::async``` 的底层实现: ```std::promise``` ,可以用这个手动创建线程
4. ```std::future``` 为了三五法则，删除了拷贝/赋值函数，如果要浅拷贝，共享一个future的话，可以用```std::shared_future```
   
## 3 互斥量
1. ```std::mutex``` 上锁调用lock(), 解锁调用unlock()
2. ```std::lock_guard``` 把锁视为资源，构造中lock，解构中unlock
3. ```std::unique_lock``` 可以调用unlock提前解构
   * 有一个额外的参数```std::defer_lock```, 指定了后```std::unique_lock```不会在构造函数中lock，要手动
3. 多个对象，每个给一个锁就可以
4. 看看锁了没 锁可以调用```try_lock()```,```try_lock_for()```,```try_lock_until()```，返回true或false
5. 上面的```try_lock()```是mutex直接调用的，```std::unique_lock```可以用```std::try_to_lock()```做参数
   * 和无参数相比，他会调用mtx.try_lock()而不是lock()，可以用grd.owns_lock()判断是否上锁成功
注：mtx是个```std::mutex```变量，grd是个```std::unique_lock```变量
   * ```std::unique_lock```还可以用```std::adopt_lock()```做参数，含义是，mtx之前已经上过锁了

## 4 死锁（dead-lock）
### 什么是死锁问题？
   由于同时执行两个线程，双方都在等待对方释放锁，但是因为等待无法释放锁
### 解决死锁问题
1. 一个线程不要同时持有两个锁
2. 保证双方上锁顺序一致
3. 用std::lock同时对多个上锁,后面再unlock()
```C++
std::lock(mtx1, mtx2);
```
4. ```std::lock_guard```相对应的```std::lock```也有RAII版本的，std::scoped_lock可以对多个mutex上锁
```C++
std::scope_lock grd(mtx1, mtx2);
```
### 解决同一线程重复调用lock()造成的死锁问题
```C++
#include <mutex>
std::mutex mtx1;
void other() {
    mtx1.lock();
    mtx1.unlock();
}
void func() {
    mtx1.lock();
    other();
    mtx1.lock();
}
```
1. 方法1就是other里面不要上锁
2. 该用```std::recursive_mutex```
   * 优点 : 会自动判断是不是在同一个线程lock()，如果是会计数器+1，减到0才会真正解锁;
   * 缺点 : 有一定的性能损失;

## 5 数据结构
* 为什么要讨论这个？
  因为std::vector是一个堆存储，不是一个多线程安全的容器，那么多个线程同时访问同一个vector会出现data-race
* 如何解决这个？
  可以用类封装堆vector的访问，使其访问受到一个mutex的保护
* 上面的解决方案会有什么问题？
  mutex::lock()不是一个const函数
  1. mutable关键字修饰咱们的std::mutex : 逻辑上const而部分成员非const
  2. 读写锁
### 读写锁 std::shared_mutex
上锁的时候指定需求，负责调度的读写锁会帮助判断需不需要等待
#### 使用方法
1. 对mutex调用```lock()``` / ```unlock()```; ```lock_shared()``` / ```unlock_shared()```;
2. RAII思想的lock_shared : 用```std::shared_lock```针对```lock_shared()```,用```std::unique_lock()```针对```lock()```;
3. 访问者模式 : (Accessor或Viewer模式) 
```C++
// 在class的public里面加函数
Accessor access() {
   return {*this};
}
```

## 6 条件变量
线程的处理中有时需要加入条件变量，可以按需取用
1. 等待被唤醒
```C++
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>

int main(){
   std::condition_variable cv;
   std::mutex mtx;
   std::thread t1([&](){
      std::unique_lock lck(mtx);
      cv.wait(lck); // 让当前线程陷入等待
      std::cout << "t1 is awake" << std::endl;
   });
   std::this_thread::sleep_for(std::chrono::milliseconds(400));
   std::cout << "notifying ..." << std::endl;
   cv.notify_one(); // 唤醒陷入等待的线程
   t1.join();
   return 0;
}
```
注: ```std::condition_variable``` 必须和 ```std::unique_lock<std::mutex>```一起使用
```std::condition_variable``` 仅仅支持 ````std::unique_lock<std::mutex>``` 作为 wait 的参数，如果需要用其他类型的 mutex 锁，可以用 ```std::condition_variable_any```。

2. 等待某一条件成真
```C++
cv.wait(lck, [&] { return ready })
```
额外指定一个参数，变成```cv.wait(lck, expr)```的形式，其中expr是一个lambda表达式，只有返回值是true时才会真正唤醒
3. 多个等待者
可以用```cv.notify_one()```和```cv.notify_all()```

## 7 原子操作

### 为什么需要原子操作
eg. ```counter += 1``` 这个在CPU看来是三条指令
1. 读取counter到rax寄存器
2. rax 寄存器的值 +1
3. 把rax写入counter变量
编译器优化，CPU为了高效，会把一条汇编指令拆分成很多微指令(micro-ops)
我们当然可以用加锁来解决多线程调用修改counter的问题，但是，mutex太重量级，会让线程挂起，从而需要通过系统调用，进入内核层，调度到其他线程执行
### 解决办法
```C++
std::atomic<int> counter = 0;
```
原子操作的汇编语言是```lock xadd %eax, (%rdx)```
注意: 用+=或++，而不是+和+分开（不能保证原子性）；

* ```counter.fetch_add(1)``` 等价于 +=1 返回的是旧值
* ```counter.store(0)``` 赋值
* ```counter.load()``` 读取值
* ```counter.exchange(val)``` 把val的值写入原子变量，返回旧值
* ```counter.compare_exchange_strong(old,val)``` 读取，比较是否相等，不相等写入，返回值为bool