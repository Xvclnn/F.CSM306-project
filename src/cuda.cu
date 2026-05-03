/*
 * cuda.cu  –  GPU дээрх merge sort
 *
 * Зорилго:
 *   tasksys.h-д тодорхойлсон TaskSystem интерфэйсийг хадгалан, GPU
 *   дээр merge sort-г bottom-up болон co-rank гэсэн хоёр түвшинд
 *   параллельчлэн хэрэгжүүлэх.
 *
 * Үндсэн санаа (Lec6 - GPU Architecture & CUDA, Lec7 - Data-parallel
 * Thinking, Kirk & Hwu Ch.12 - Merge):
 *
 *   1. Bottom-up merge sort:
 *      width = 1, 2, 4, ..., n/2 хүртэл явна.
 *      width бүрт массивыг 2*width хэмжээтэй хосуудад хувааж merge хийнэ.
 *
 *   2. Хоёр төрлийн kernel:
 *      a) simple_merge_kernel – нэг thread нэг хосыг бүхэлд нь merge
 *         хийнэ. Хосын хэмжээ жижиг, тоо нь их үед (эхний түвшинүүд)
 *         энэ нь GPU-г бүрэн хангалттай ачаалдаг.
 *      b) corank_merge_kernel – нэг хосыг олон thread хамтдаа merge
 *         хийнэ. Хосын хэмжээ том, тоо нь цөөн үед (сүүлийн түвшинүүд)
 *         нэг хос дотор parallelism үүсгэхийн тулд co-rank ашиглана.
 *
 *   3. Хэмжилт:
 *      H2D, kernel, D2H тус бүрийн хугацааг chrono-р нарийн хэмжиж,
 *      нийт data_transfer_time_ms болон data_transferred_bytes-ийг
 *      tasksys.h-д тодорхойлсон public талбарт хадгална.
 *
 * Ашиглах:
 *   nvcc -O3 -std=c++17 -DUSE_CUDA -Xcompiler -fopenmp \
 *        src/main.cpp src/tasksys.cpp src/cuda.cu -o src/main -lgomp
 */

#include "tasksys.h"
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

using hrc = std::chrono::high_resolution_clock;
using ms  = std::chrono::duration<double, std::milli>;

// ----------------------------------------------------------------
// CUDA алдаа шалгах макро.
// Алдаа гарвал шууд stderr-д бичээд програмыг зогсоох. Ингэснээр
// CSV-д хагас үр дүн бичигдэхгүй – туршилтын тогтвортой байдлыг хангана.
// ----------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d %s -> %s\n",                   \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err__));     \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)


// ----------------------------------------------------------------
// Co-rank device функц (Kirk & Hwu Fig 12.5)
//
// Гаралтын C массивын k-р индекст A-аас хэдэн элемент авсныг
// binary search-р O(log N)-д олно.  i + j == k invariant.
//
// Шинж чанар:
//   A[i-1] > B[j]  -> A-аас хэт олон авсан, i-г бууруулна
//   B[j-1] > A[i]  -> B-ээс хэт олон авсан, j-г бууруулна (i-г нэмэгдүүлнэ)
//   Хоёулаа биелсэн -> олдсон, i-г буцаана
// ----------------------------------------------------------------
__device__ static int d_co_rank(int k,
                                const float* A, int m,
                                const float* B, int n)
{
    int low  = (k > n) ? (k - n) : 0;
    int high = (k < m) ? k       : m;

    while (low <= high) {
        int i = (low + high) >> 1;
        int j = k - i;

        if (i > 0 && j < n && A[i - 1] > B[j]) {
            high = i - 1;
        } else if (j > 0 && i < m && B[j - 1] > A[i]) {
            low = i + 1;
        } else {
            return i;
        }
    }
    return low;
}


// ----------------------------------------------------------------
// simple_merge_kernel
//
// Bottom-up merge sort-ын нэг түвшнийг гүйцэтгэнэ.
// Нэг thread яг нэг 2*width хэмжээтэй хосыг бүхэлд нь хариуцна.
//
//   in[]  – оролтын массив (хос бүр дотроо эрэмбэлэгдсэн)
//   out[] – гаралтын массив (нэгтгэсний дараах эрэмбэлэгдсэн утгууд)
//
// Жижиг хосын үед (эхний түвшинүүд) thread-ийн тоо их учраас
// энэ хувилбар маш сайн ачааллыг түгээдэг.
// ----------------------------------------------------------------
__global__ void simple_merge_kernel(const float* __restrict__ in,
                                    float* __restrict__       out,
                                    int                       n,
                                    int                       width)
{
    int pair_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_start = pair_idx * 2 * width;
    if (pair_start >= n) return;

    int mid      = min(pair_start + width,     n);
    int pair_end = min(pair_start + 2 * width, n);

    int i = pair_start, j = mid, k = pair_start;
    while (i < mid && j < pair_end) {
        if (in[i] <= in[j]) out[k++] = in[i++];
        else                out[k++] = in[j++];
    }
    while (i < mid)      out[k++] = in[i++];
    while (j < pair_end) out[k++] = in[j++];
}


// ----------------------------------------------------------------
// corank_merge_kernel
//
// Нэг 2*width хэмжээтэй хосыг олон thread зэрэг merge хийнэ.
// Энэ нь width том болсон үе (сүүлийн түвшинүүд) ашигтай:
// тэр үед хосын тоо цөөрч, simple_merge_kernel ашиглавал
// thread-уудын тоо хэт цөөрч GPU дутуу ачаалагдана.
//
// segment_size – нэг thread хариуцах гаралтын элементийн тоо.
// Зэрэгцүүлэлт: pair_size (2*width) нь segment_size-ийн үржвэр байх ёстой
// (бид зөвхөн pair_size > 1024 үед энэ kernel-ийг дуудах тул бүх
// width >= 512 тохиолдолд биелнэ; segment_size = 64).
// ----------------------------------------------------------------
__global__ void corank_merge_kernel(const float* __restrict__ in,
                                    float* __restrict__       out,
                                    int                       n,
                                    int                       width,
                                    int                       segment_size)
{
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    int k_start = tid * segment_size;
    if (k_start >= n) return;
    int k_end = min(k_start + segment_size, n);

    int pair_size  = 2 * width;
    int pair_idx   = k_start / pair_size;
    int pair_start = pair_idx * pair_size;
    int mid        = min(pair_start + width,     n);
    int pair_end   = min(pair_start + pair_size, n);

    // Нэг thread зөвхөн нэг хосын дотор ажиллана. Шаардлагатай бол
    // хосын төгсгөл хүртэл хязгаарлана.
    if (k_end > pair_end) k_end = pair_end;

    // Хосын дотоод координат руу шилжүүлнэ.
    int local_k_start = k_start - pair_start;
    int local_k_end   = k_end   - pair_start;

    const float* A = in + pair_start;
    int          m = mid - pair_start;
    const float* B = in + mid;
    int          n_b = pair_end - mid;

    // Co-rank-р оролтын муж тогтооно.
    int i_start = d_co_rank(local_k_start, A, m, B, n_b);
    int j_start = local_k_start - i_start;
    int i_end   = d_co_rank(local_k_end,   A, m, B, n_b);
    int j_end   = local_k_end   - i_end;

    // Хариуцсан хэсгээ sequential merge хийнэ.
    int i = i_start, j = j_start, k = k_start;
    while (i < i_end && j < j_end) {
        if (A[i] <= B[j]) out[k++] = A[i++];
        else              out[k++] = B[j++];
    }
    while (i < i_end) out[k++] = A[i++];
    while (j < j_end) out[k++] = B[j++];
}


// ================================================================
// TaskSystemCUDA::run_sort
//
// Алхамууд:
//   1) Device санах ой захиалах (double-buffer: d_in, d_out)
//   2) H2D дамжуулалт хийж хугацааг хэмжих
//   3) Bottom-up merge sort: width = 1, 2, 4, ...
//        - Хосын хэмжээ <= 1024  -> simple_merge_kernel
//        - Хосын хэмжээ >  1024  -> corank_merge_kernel
//      Үе бүрийн дараа d_in, d_out-г swap хийнэ.
//   4) D2H дамжуулалт хийж хугацааг хэмжих
//   5) Device санах ой чөлөөлөх
//   6) data_transfer_time_ms = H2D + D2H
//      data_transferred_bytes = 2 * n * sizeof(float)
//
// num_threads параметр нь интерфэйст шаардсан тул үлдээсэн –
// GPU дээр launch config-г bot хариуцна.
// ================================================================
void TaskSystemCUDA::run_sort(int /*num_threads*/, float* h_array, int array_size)
{
    if (h_array == nullptr || array_size <= 0) return;

    // Хэмжилтийн талбаруудыг тэглэнэ
    data_transfer_time_ms  = 0.0;
    data_transferred_bytes = 0;

    size_t bytes = static_cast<size_t>(array_size) * sizeof(float);

    float* d_in  = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_in),  bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes));

    // ---------- 1. Host -> Device ---------------------------------
    auto h2d_t0 = hrc::now();
    CUDA_CHECK(cudaMemcpy(d_in, h_array, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto h2d_t1 = hrc::now();
    double h2d_ms = ms(h2d_t1 - h2d_t0).count();

    // ---------- 2. Bottom-up merge sort --------------------------
    const int BLOCK            = 256;     // нэг block-ийн thread тоо
    const int CORANK_PAIR_LIM  = 1024;    // pair_size энээс их үед corank
    const int CORANK_SEGMENT   = 64;      // corank-ийн нэг thread-ийн ачаалал

    for (int width = 1; width < array_size; width *= 2) {
        int pair_size = 2 * width;

        if (pair_size <= CORANK_PAIR_LIM) {
            // Олон жижиг хос – нэг хос нэг thread
            int num_pairs = (array_size + pair_size - 1) / pair_size;
            int grid      = (num_pairs + BLOCK - 1) / BLOCK;
            simple_merge_kernel<<<grid, BLOCK>>>(d_in, d_out, array_size, width);
        } else {
            // Цөөн том хос – нэг хосыг олон thread зэрэг гүйцэтгэнэ
            int num_threads = (array_size + CORANK_SEGMENT - 1) / CORANK_SEGMENT;
            int grid        = (num_threads + BLOCK - 1) / BLOCK;
            corank_merge_kernel<<<grid, BLOCK>>>(d_in, d_out, array_size,
                                                 width, CORANK_SEGMENT);
        }

        // Хэрэв kernel-ийн launch алдаа гарвал шууд илрүүлнэ.
        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess) {
            fprintf(stderr, "[CUDA KERNEL ERROR] width=%d: %s\n",
                    width, cudaGetErrorString(kerr));
            std::exit(1);
        }

        // Double buffer swap: дараагийн түвшинд d_in нь нэгтгэсэн утгууд
        std::swap(d_in, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- 3. Device -> Host ---------------------------------
    auto d2h_t0 = hrc::now();
    CUDA_CHECK(cudaMemcpy(h_array, d_in, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto d2h_t1 = hrc::now();
    double d2h_ms_v = ms(d2h_t1 - d2h_t0).count();

    // ---------- 4. Цэвэрлэгээ -------------------------------------
    cudaFree(d_in);
    cudaFree(d_out);

    // ---------- 5. Хэмжилт хадгалах -------------------------------
    data_transfer_time_ms  = h2d_ms + d2h_ms_v;
    data_transferred_bytes = static_cast<long long>(2) * static_cast<long long>(bytes);
}
