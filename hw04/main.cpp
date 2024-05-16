#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

#define N 42
// 1.变成内联函数
static float frand() {
    return (float)rand() / RAND_MAX * 2 - 1;
}
// 2.改AOS为SOA
struct Star {
    float px[N];
    float py[N];
    float pz[N];
    float vx[N];
    float vy[N];
    float vz[N];
    float mass[N];
};

Star stars;

void init() {
    #pragma GCC unroll 4
        for (int i = 0; i < N - 2; i++) {
            stars.px[i] = frand();
            stars.py[i] = frand();
            stars.pz[i] = frand();
            stars.vx[i] = frand();
            stars.vy[i] = frand();
            stars.vz[i] = frand();
            stars.mass[i] = frand() + 1;
        }
    for (int i = N - 2; i < N; i++) {
        stars.px[i] = frand();
        stars.py[i] = frand();
        stars.pz[i] = frand();
        stars.vx[i] = frand();
        stars.vy[i] = frand();
        stars.vz[i] = frand();
        stars.mass[i] = frand() + 1;
    }
}

float G = 0.001;
float eps = 0.001;
float dt = 0.01;

// 3.处理一下数值计算
void step() {
    float eps2 = eps * eps;
    #pragma GCC unroll 4
        for (int i = 0; i < N - 2; i++) {
            for (int j = 0; j < N; j++) {
                float dx = stars.px[j] - stars.px[i];
                float dy = stars.py[j] - stars.py[i];
                float dz = stars.pz[j] - stars.pz[i];
                float d2 = dx * dx + dy * dy + dz * dz + eps2;
                d2 *= std::sqrt(d2);
                d2 = 1 / d2;
                float tmp = stars.mass[j] * G * dt * d2;
                stars.vx[i] += dx * tmp;
                stars.vy[i] += dy * tmp;
                stars.vz[i] += dz * tmp;
            }
        }
    for (int i = N -2 ; i < N ; i++) {
        for (int j = 0; j < N; j++) {
            float dx = stars.px[j] - stars.px[i];
            float dy = stars.py[j] - stars.py[i];
            float dz = stars.pz[j] - stars.pz[i];
            float d2 = dx * dx + dy * dy + dz * dz + eps2;
            d2 *= std::sqrt(d2);
            d2 = 1 / d2;
            float tmp = stars.mass[j] * G * dt * d2;
            stars.vx[i] += dx * tmp;
            stars.vy[i] += dy * tmp;
            stars.vz[i] += dz * tmp;
        }
    }
    #pragma omp simd
        for (int i = 0; i < N - 2; i++) {
            stars.px[i] += stars.vx[i] * dt;
            stars.py[i] += stars.vy[i] * dt;
            stars.pz[i] += stars.vz[i] * dt;
        }
    for (int i = N -2 ; i < N ; i++) {
        stars.px[i] += stars.vx[i] * dt;
        stars.py[i] += stars.vy[i] * dt;
        stars.pz[i] += stars.vz[i] * dt;
    }
}

float calc() {
    float energy = 0;
    #pragma omp simd
        for (int i = 0; i < N - 2; i++) {
            float v2 = stars.vx[i] * stars.vx[i] + stars.vy[i] * stars.vy[i] + stars.vz[i] * stars.vz[i];
            energy += stars.mass[i] * v2 / 2;
            for (int j = 0; j < N; j++) {
                float dx = stars.px[j] - stars.px[i];
                float dy = stars.py[j] - stars.py[i];
                float dz = stars.pz[j] - stars.pz[i];
                float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
                energy -= stars.mass[j] * stars.mass[i] * G / std::sqrt(d2) / 2;
            }
        }
    for (int i = N -2 ; i < N ; i++) {
        float v2 = stars.vx[i] * stars.vx[i] + stars.vy[i] * stars.vy[i] + stars.vz[i] * stars.vz[i];
        energy += stars.mass[i] * v2 / 2;
        for (int j = 0; j < N; j++) {
            float dx = stars.px[j] - stars.px[i];
            float dy = stars.py[j] - stars.py[i];
            float dz = stars.pz[j] - stars.pz[i];
            float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
            energy -= stars.mass[j] * stars.mass[i] * G / std::sqrt(d2) / 2;
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

// 优化最后结果大概是 130 ms