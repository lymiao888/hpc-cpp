#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
static constexpr int num = 48;
static float frand() {
    return (float)rand() / RAND_MAX * 2 - 1;
}
//我觉得这里可以改结构体！
struct alignas(16) Star {
    float px[4], py[4], pz[4];
    float vx[4], vy[4], vz[4];
    float mass[4];
};

std::vector<Star> stars;

//这里的for循环影响不大，主要跟结构体有关
void init() {
    for (int i = 0; i < num / 4; i++) {
        for(int j = 0; j < 4 ; i++){
            stars[i].px[j] = frand();
            stars[i].py[j] = frand();
            stars[i].pz[j] = frand();
            stars[i].vx[j] = frand();
            stars[i].vy[j] = frand();
            stars[i].vz[j] = frand();
            stars[i].mass[j] = frand() + 1;
        }
    }
}

static const float G = 0.001;
static const float eps = 0.001;
static const float dt = 0.01;

void step() {
    for (auto &star: stars) {
        for (auto &other: stars) {
            float dx = other.px - star.px;
            float dy = other.py - star.py;
            float dz = other.pz - star.pz;
            float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
            d2 *= sqrt(d2);
            star.vx += dx * other.mass * G * dt / d2;
            star.vy += dy * other.mass * G * dt / d2;
            star.vz += dz * other.mass * G * dt / d2;
        }
    }
    for (auto &star: stars) {
        star.px += star.vx * dt;
        star.py += star.vy * dt;
        star.pz += star.vz * dt;
    }
}

float calc() {
    float energy = 0;
    for (auto &star: stars) {
        float v2 = star.vx * star.vx + star.vy * star.vy + star.vz * star.vz;
        energy += star.mass * v2 / 2;
        for (auto &other: stars) {
            float dx = other.px - star.px;
            float dy = other.py - star.py;
            float dz = other.pz - star.pz;
            float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
            energy -= other.mass * star.mass * G / std::sqrt(d2) / 2;
        }
    }
    return energy;
}

template <class Func>
long benchmark(Func const &func) {
    auto t0 = std::chrono::steady_clock::now();
    func();
    auto t1 = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    return dt.count();
}

int main() {
    init();
    printf("Initial energy: %f\n", calc());
    auto dt = benchmark([&] {
        for (int i = 0; i < 100000; i++)
            step();
    });
    printf("Final energy: %f\n", calc());
    printf("Time elapsed: %ld ms\n", dt);
    return 0;
}
