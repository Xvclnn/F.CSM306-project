/*
 * cuda.cu  –  GPU merge sort (bottom-up, array-based)
 *
 * Lec6 – GPU Architecture & CUDA Programming:
 *   Linked list нь pointer-р холбогдсон тул GPU-д шууд ашиглах
 *   боломжгүй (адресс тооцоолол хэт нарийн). Иймд:
 *     1) List-ийн утгуудыг host array-д хуулна
 *     2) cudaMemcpy-р device руу илгээнэ   (H->D)
 *     3) Bottom-up merge sort kernel-г ажиллуулна
 *     4) Эрэмбэлэгдсэн array-г host руу хуулна  (D->H)
 *     5) Утгуудыг list node-уудад буцааж бичнэ
 *
 * Kernel загвар (Lec6: thread hierarchy):
 *   - Grid  : (pairs + BLOCK - 1) / BLOCK blocks
 *   - Block : BLOCK = 256 threads
 *   - Thread: нэг merge pair-г хариуцна
 *
 * Тайлбар хэмжилтүүд (h2d_ms, kernel_ms, d2h_ms) нь тайланд
 * "Data transfer time" болон "Achievable performance"-ыг тооцоолоход
 * ашиглагдана.
 */

#include "tasksys.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

using hrc = std::chrono::high_resolution_clock;
using ms  = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------
// Device function: arr[left..mid-1] ба arr[mid..right-1]-г нэгтгэнэ
// Үр дүнг temp-д бичиж, arr руу хуулна
// ---------------------------------------------------------------
__device__ static void deviceMerge(int* arr, int* temp,
                                    int left, int mid, int right) {
    int i = left, j = mid, k = left;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else                  temp[k++] = arr[j++];
    }
    while (i < mid)   temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];

    for (int x = left; x < right; x++)
        arr[x] = temp[x];
}

// ---------------------------------------------------------------
// Kernel: bottom-up merge sort-ын нэг давхрага
//   width – одоогийн sub-array-ийн хагас урт (1, 2, 4, 8, ...)
//   Нэг thread нэг merge хийнэ.
// ---------------------------------------------------------------
__global__ void mergeSortKernel(int* arr, int* temp, int n, int width) {
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int left  = tid * 2 * width;

    if (left >= n) return;

    int mid   = min(left + width,     n);
    int right = min(left + 2 * width, n);

    if (mid < right)
        deviceMerge(arr, temp, left, mid, right);
}

// ---------------------------------------------------------------
// TaskSystemCUDA::sort
// ---------------------------------------------------------------
Node* TaskSystemCUDA::sort(Node* head) {
    // Нийт node тоолох
    int n = 0;
    for (Node* p = head; p != nullptr; p = p->next)
        n++;

    if (n <= 1) return head;

    // List-ийн утгуудыг host array-д хуулах
    int* h_arr = new int[n];
    {
        Node* p = head;
        for (int i = 0; i < n; i++, p = p->next)
            h_arr[i] = p->val;
    }

    // Device санах ой захиалах
    int* d_arr  = nullptr;
    int* d_temp = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_arr),  n * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_temp), n * sizeof(int));

    // --- Host -> Device дамжуулалт (хугацааг хэмжинэ) ----------
    auto t0 = hrc::now();
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    auto t1 = hrc::now();
    h2d_ms = ms(t1 - t0).count();

    // --- GPU kernel: bottom-up merge sort -----------------------
    // Lec6: block size 256 = 8 warp, SM-д дор хаяж нэг block байна
    const int BLOCK = 256;

    auto t2 = hrc::now();
    for (int width = 1; width < n; width *= 2) {
        int pairs    = (n + 2 * width - 1) / (2 * width);
        int gridSize = (pairs + BLOCK - 1) / BLOCK;
        mergeSortKernel<<<gridSize, BLOCK>>>(d_arr, d_temp, n, width);
    }
    cudaDeviceSynchronize();
    auto t3 = hrc::now();
    kernel_ms = ms(t3 - t2).count();

    // --- Device -> Host дамжуулалт (хугацааг хэмжинэ) ----------
    auto t4 = hrc::now();
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto t5 = hrc::now();
    d2h_ms = ms(t5 - t4).count();

    cudaFree(d_arr);
    cudaFree(d_temp);

    // Эрэмбэлэгдсэн утгуудыг list node-уудад буцааж бичнэ
    {
        Node* p = head;
        for (int i = 0; i < n; i++, p = p->next)
            p->val = h_arr[i];
    }

    delete[] h_arr;
    return head;
}
