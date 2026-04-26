#include "tasksys.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cassert>

// -----------------------------------------------------------
// Linked-list utilities
// -----------------------------------------------------------

static Node* buildList(const std::vector<int>& data) {
    Node* head = nullptr;
    Node* tail = nullptr;
    for (int v : data) {
        Node* n = new Node(v);
        if (!tail) { head = tail = n; }
        else       { tail->next = n; tail = n; }
    }
    return head;
}

static void freeList(Node* head) {
    while (head) {
        Node* next = head->next;
        delete head;
        head = next;
    }
}

// Verify list is sorted (debug check, not timed)
static bool isSorted(Node* head) {
    while (head && head->next) {
        if (head->val > head->next->val) return false;
        head = head->next;
    }
    return true;
}

// -----------------------------------------------------------
// Benchmark: build list, sort, measure wall time (ms)
// -----------------------------------------------------------
static double benchmark(ITaskSystem* sys,
                        const std::vector<int>& data,
                        const std::string& label,
                        int size)
{
    Node* list = buildList(data);

    auto t1 = std::chrono::high_resolution_clock::now();
    Node* sorted = sys->sort(list);
    auto t2 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Correctness check (only on small inputs to keep output clean)
    if (size <= 100000) {
        assert(isSorted(sorted) && "Sort result is incorrect!");
    }

    freeList(sorted);
    return ms;
}

// -----------------------------------------------------------
// main
// -----------------------------------------------------------
int main() {
    const std::vector<int> sizes = {10000, 100000, 1000000};
    const unsigned int SEED = 42;

    std::ofstream csv("./csv/output.csv");
    if (!csv.is_open()) {
        std::cerr << "Cannot open ./csv/output.csv\n";
        return 1;
    }
    csv << "method,input_size,time_ms\n";

    // Instantiate all CPU systems once (constructor reads hw concurrency)
    TaskSystemSerial  serial;
    TaskSystemThread  threaded;
    TaskSystemOpenMP  openmp_sys;
#ifdef USE_CUDA
    TaskSystemCUDA    cuda_sys;
#endif

    for (int sz : sizes) {
        // Generate identical random input for every method
        std::mt19937 rng(SEED);
        std::uniform_int_distribution<int> dist(0, 1000000);
        std::vector<int> data(sz);
        for (auto& v : data) v = dist(rng);

        std::cout << "\n=== Input size: " << sz << " ===\n";

        // 1. Sequential
        double t1 = benchmark(&serial, data, "sequential", sz);
        std::cout << "  sequential  : " << t1 << " ms\n";
        csv << "sequential," << sz << "," << t1 << "\n";

        // 2. std::thread
        double t2 = benchmark(&threaded, data, "std_thread", sz);
        std::cout << "  std_thread  : " << t2 << " ms\n";
        csv << "std_thread," << sz << "," << t2 << "\n";

        // 3. OpenMP
        double t3 = benchmark(&openmp_sys, data, "openmp", sz);
        std::cout << "  openmp      : " << t3 << " ms\n";
        csv << "openmp," << sz << "," << t3 << "\n";

        // 4. CUDA (only when compiled with -DUSE_CUDA)
#ifdef USE_CUDA
        double t4 = benchmark(&cuda_sys, data, "cuda", sz);
        std::cout << "  cuda        : " << t4 << " ms\n";
        csv << "cuda," << sz << "," << t4 << "\n";
#else
        std::cout << "  cuda        : N/A (not compiled)\n";
        csv << "cuda," << sz << ",N/A\n";
#endif
    }

    csv.close();
    std::cout << "\nResults saved to ./csv/output.csv\n";
    return 0;
}
