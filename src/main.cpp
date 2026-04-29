#include "tasksys.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

using hrc = std::chrono::high_resolution_clock;
using ms  = std::chrono::duration<double, std::milli>;

// ================================================================
// Linked list туслах функцүүд
// ================================================================

static Node* buildList(const std::vector<int>& data) {
    Node* head = nullptr;
    Node* tail = nullptr;
    for (int v : data) {
        Node* nd = new Node(v);
        if (tail == nullptr) { head = tail = nd; }
        else { tail->next = nd; tail = nd; }
    }
    return head;
}

static void freeList(Node* head) {
    while (head != nullptr) {
        Node* nxt = head->next;
        delete head;
        head = nxt;
    }
}

// O(n) дараалсан шалгалт – sort бүрийн дараа явуулна
static bool isSorted(Node* head) {
    while (head != nullptr && head->next != nullptr) {
        if (head->val > head->next->val) return false;
        head = head->next;
    }
    return true;
}

// ================================================================
// Benchmark: жагсаалт барьж, sort хийж, хугацаа хэмжинэ.
// I/O болон жагсаалт барих хугацааг оруулахгүй.
// ================================================================
static double benchmark(ITaskSystem* sys, const std::vector<int>& data,
                        const char* label) {
    Node* list = buildList(data);

    auto t1 = hrc::now();
    Node* sorted = sys->sort(list);
    auto t2 = hrc::now();

    double elapsed = ms(t2 - t1).count();

    // Бүх хэмжээнд зөв эрэмбэлэгдсэн эсэхийг шалгана
    if (!isSorted(sorted)) {
        fprintf(stderr, "[ERROR] %s: sort result is WRONG!\n", label);
    }

    freeList(sorted);
    return elapsed;
}

// ================================================================
// Системийн мэдээлэл хэвлэх (Linux болон macOS)
// ================================================================
static void printSysInfo() {
    printf("========================================\n");
    printf("  Системийн мэдээлэл\n");
    printf("========================================\n");

#ifdef __linux__
    // CPU нэр
    FILE* f = popen("grep 'model name' /proc/cpuinfo | head -1 "
                    "| sed 's/.*: //'", "r");
    if (f) {
        char buf[256] = {};
        if (fgets(buf, sizeof(buf), f)) {
            buf[strcspn(buf, "\n")] = '\0';
            printf("CPU     : %s\n", buf);
        }
        pclose(f);
    }
    // Нийт логик цөм
    FILE* g = popen("nproc", "r");
    if (g) {
        int ncpu = 0;
        fscanf(g, "%d", &ncpu);
        pclose(g);
        printf("Цөм     : %d\n", ncpu);
    }
    // RAM хэмжээ
    FILE* h = popen("grep 'MemTotal' /proc/meminfo | awk '{print $2}'", "r");
    if (h) {
        long mem_kb = 0;
        fscanf(h, "%ld", &mem_kb);
        pclose(h);
        printf("RAM     : %.1f GB\n", mem_kb / 1048576.0);
    }
#else
    // macOS
    FILE* f = popen("sysctl -n machdep.cpu.brand_string", "r");
    if (f) {
        char buf[256] = {};
        if (fgets(buf, sizeof(buf), f)) {
            buf[strcspn(buf, "\n")] = '\0';
            printf("CPU     : %s\n", buf);
        }
        pclose(f);
    }
#endif

    printf("HW threads: %u\n", std::thread::hardware_concurrency());

#ifdef _OPENMP
    printf("OMP threads: %d\n", omp_get_max_threads());
#else
    printf("OMP threads: N/A (OpenMP compile-д ороогүй)\n");
#endif

    printf("========================================\n\n");
}

// ================================================================
// main
// ================================================================
int main() {
    printSysInfo();

    // Туршилтын өгөгдлийн хэмжээнүүд
    const std::vector<int> sizes = {10000, 100000, 1000000};
    const unsigned int SEED = 42;

    // CSV файлыг нээнэ
    std::ofstream csv("./csv/output.csv");
    if (!csv.is_open()) {
        fprintf(stderr, "[ERROR] ./csv/output.csv нээгдсэнгүй\n");
        return 1;
    }

    // Баганы толгой:
    // time_ms       – нийт ажиллах хугацаа
    // speedup       – sequential-тай харьцуулсан хурдац (Amdahl's Law)
    // total_ops     – merge sort-ын онолын харьцуулалтын тоо ≈ n*log2(n)
    // h2d_ms        – CUDA: host->device дамжуулалтын хугацаа
    // kernel_ms     – CUDA: GPU kernel-ийн цэвэр хугацаа
    // d2h_ms        – CUDA: device->host дамжуулалтын хугацаа
    csv << "method,input_size,time_ms,speedup,total_ops,"
           "h2d_ms,kernel_ms,d2h_ms\n";

    // Sort системүүдийг нэг удаа үүсгэнэ
    TaskSystemSerial  serial;
    TaskSystemThread  threaded;
    TaskSystemOpenMP  omp_sys;
#ifdef USE_CUDA
    TaskSystemCUDA    cuda_sys;
#endif

    for (int sz : sizes) {
        // Бүх аргад ижил санамсаргүй оролт ашиглана
        std::mt19937 rng(SEED);
        std::uniform_int_distribution<int> dist(0, 1000000);
        std::vector<int> data(sz);
        for (int& v : data) v = dist(rng);

        // Merge sort-ын онолын харьцуулалтын тоо ≈ n * log2(n)
        long long total_ops = static_cast<long long>(
            std::round(sz * std::log2(sz)));

        printf("--- Оролтын хэмжээ: %d элемент ---\n", sz);

        // 1. Sequential – суурь хугацаа (speedup тооцоолох эталон)
        double t_seq = benchmark(&serial, data, "sequential");
        printf("  sequential : %8.3f ms  (SpeedUp = 1.00)\n", t_seq);
        csv << "sequential," << sz << "," << t_seq << ",1.00,"
            << total_ops << ",0,0,0\n";

        // 2. std::thread
        double t_thr = benchmark(&threaded, data, "std_thread");
        double su_thr = t_seq / t_thr;
        printf("  std_thread : %8.3f ms  (SpeedUp = %.2f)\n", t_thr, su_thr);
        csv << "std_thread," << sz << "," << t_thr << ","
            << su_thr << "," << total_ops << ",0,0,0\n";

        // 3. OpenMP
        double t_omp = benchmark(&omp_sys, data, "openmp");
        double su_omp = t_seq / t_omp;
        printf("  openmp     : %8.3f ms  (SpeedUp = %.2f)\n", t_omp, su_omp);
        csv << "openmp," << sz << "," << t_omp << ","
            << su_omp << "," << total_ops << ",0,0,0\n";

        // 4. CUDA
#ifdef USE_CUDA
        double t_cuda = benchmark(&cuda_sys, data, "cuda");
        double su_cuda = t_seq / t_cuda;
        printf("  cuda       : %8.3f ms  (SpeedUp = %.2f)\n", t_cuda, su_cuda);
        printf("               H->D: %.3f ms | kernel: %.3f ms | D->H: %.3f ms\n",
               cuda_sys.h2d_ms, cuda_sys.kernel_ms, cuda_sys.d2h_ms);
        csv << "cuda," << sz << "," << t_cuda << "," << su_cuda << ","
            << total_ops << "," << cuda_sys.h2d_ms << ","
            << cuda_sys.kernel_ms << "," << cuda_sys.d2h_ms << "\n";
#else
        printf("  cuda       : N/A (USE_CUDA compile flag байхгүй)\n");
        csv << "cuda," << sz << ",0,0," << total_ops << ",0,0,0\n";
#endif

        printf("\n");
    }

    csv.close();
    printf("Үр дүн хадгалагдлаа: ./csv/output.csv\n");
    return 0;
}
