// 小彭老师作业05：假装是多线程 HTTP 服务器 - 富连网大厂面试官觉得很赞
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <shared_mutex>
#include <sstream>
#include <cstdlib>
#include <string>
#include <thread>
#include <map>
#include <mutex>
#include <vector>


struct User {
    std::string password;
    std::string school;
    std::string phone;
};

std::map<std::string, User> users;
std::shared_mutex mtx_users;

std::map<std::string, std::chrono::steady_clock::time_point> has_login;  // 换成 std::chrono::seconds 之类的
std::shared_mutex mtx_login;

// 作业要求1：把这些函数变成多线程安全的
// 提示：能正确利用 shared_mutex 加分，用 lock_guard 系列加分
// 这三个函数分别是对两个变量读写，对一个变量读，对一个变量写
std::string do_register(std::string username, std::string password, std::string school, std::string phone) {
    std::shared_lock grd1(mtx_login);
    std::unique_lock grd2(mtx_users);
    User user = {password, school, phone};
    if (users.emplace(username, user).second)
        return "注册成功";
    else
        return "用户名已被注册";
}

std::string do_login(std::string username, std::string password) {
    std::unique_lock grd1(mtx_login); 
    std::shared_lock grd2(mtx_users);
    auto now = std::chrono::steady_clock::now(); 
    if (has_login.find(username) != has_login.end()) {
        int64_t sec = std::chrono::duration_cast<std::chrono::seconds>(now - has_login.at(username)).count(); 
        return std::to_string(sec) + "秒内登录过";
    }
    has_login[username] = now;
    if (users.find(username) == users.end())
        return "用户名错误";
    if (users.at(username).password != password)
        return "密码错误";
    return "登录成功";
}

std::string do_queryuser(std::string username) {
    std::shared_lock grd1(mtx_login); 
    std::shared_lock grd2(mtx_users); 
    if (users.find(username) != users.end()){
        auto &user = users.at(username);
        std::stringstream ss;
        ss << "用户名: " << username << std::endl;
        ss << "学校: " << user.school << std::endl;
        ss << "电话: " << user.phone << std::endl;
        return ss.str();
    }
}

class ThreadPool {
    std::vector<std::thread> m_pool;
public:
    void push_back(std::thread thr) {
        // 作业要求3：如何让这个线程保持在后台执行不要退出？
        // 提示：改成 async 和 future 且用法正确也可以加分
        // std::thread thr(start);
        m_pool.push_back(std::move(thr));
    }
    ~ThreadPool(){
        for(auto &t: m_pool) t.join();
    }
};

ThreadPool tpool;


namespace test {  // 测试用例？出水用力！
std::string username[] = {"张心欣", "王鑫磊", "彭于斌", "胡原名"};
std::string password[] = {"hellojob", "anti-job42", "cihou233", "reCihou_!"};
std::string school[] = {"九百八十五大鞋", "浙江大鞋", "剑桥大鞋", "麻绳理工鞋院"};
std::string phone[] = {"110", "119", "120", "12315"};
}

int main() {
    // std::future<void> frt_register = std::async([&] {
    //         std::cout << do_register(test::username[rand() % 4], test::password[rand() % 4], test::school[rand() % 4], test::phone[rand() % 4]) << std::endl;
    //     });
    // std::future<void> frt_login = std::async([&] {
    //         std::cout << do_login(test::username[rand() % 4], test::password[rand() % 4]) << std::endl;
    //     });
    // std::future<void> frt_queryuser = std::async([&] {
    //         std::cout << do_queryuser(test::username[rand() % 4]) << std::endl;
    //     });
    // for (int i = 0; i < 262144; i++) {
    //     frt_register.get();
    //     frt_login.get();
    //     frt_queryuser.get();
    // }
    for (int i = 0; i < 262144; i++) {
        tpool.push_back(std::thread([&] {
            std::cout << do_register(test::username[rand() % 4], test::password[rand() % 4], test::school[rand() % 4], test::phone[rand() % 4]) << std::endl;
        }));
        tpool.push_back(std::thread([&] {
            std::cout << do_login(test::username[rand() % 4], test::password[rand() % 4]) << std::endl;
        }));
        tpool.push_back(std::thread([&] {
            std::cout << do_queryuser(test::username[rand() % 4]) << std::endl;
        }));
    }
    // 作业要求4：等待 tpool 中所有线程都结束后再退出
    return 0;
}
