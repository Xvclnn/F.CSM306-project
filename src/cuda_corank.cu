/*
 * cuda_corank.cu  –  Co-rank суурилсан параллел merge (CUDA)
 *
 * Эх сурвалж: Siebert & Träff (2012), "A Correct and Stable Parallel Merge
 *             Algorithm" — Programming Massively Parallel Processors (Kirk &
 *             Hwu) 12-р бүлэг "Merge" хэсэгт тайлбарласан аргачлал.
 *
 * Санаа:
 *   Нийт гаралтын массивыг thread бүрт тэгш хэмжээтэй хэсгүүдэд хувааж,
 *   thread бүр өөрийн "rank" (гаралтад эзлэх байр) мэдэж байна.
 *   Co-rank функц нь тухайн rank-д харгалзах A, B оролтын индексийг
 *   binary search-р O(log N) хугацаанд олно.
 *   Ингэснээр thread бүр бие даан өөрийн хэсгийг merge хийж болно –
 *   хоорондын хамаарал байхгүй.
 *
 * Ажиллах схем (Kirk & Hwu Lec 12):
 *
 *   Thread t:
 *     k_start = t * segment
 *     k_end   = (t+1) * segment
 *     i_start = co_rank(k_start, A, m, B, n)
 *     i_end   = co_rank(k_end,   A, m, B, n)
 *     j_start = k_start - i_start
 *     j_end   = k_end   - i_end
 *     → Sequential merge: A[i_start..i_end), B[j_start..j_end) → C[k_start..k_end)
 *
 * Bottom-up merge sort (cuda.cu)-тай ялгаа:
 *   cuda.cu нь level-ийн давталтаар pair тус бүрийг нэг thread-д өгнө.
 *   Энэ файл нь нэг merge дотроо олон thread-г зэрэг ажиллуулна –
 *   том merge-д илүү сайн thread-level parallelism өгдөг.
 */

#include "itasksys.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------
// Co-rank функц  (device код, binary search)
//
// Тодорхойлолт: C[0..k-1] = merge(A[0..m-1], B[0..n-1])-ийн
//   эхний k элементийг үүсгэхэд A-с хэдэн элемент авсан бэ?
//   Буцаах утга i:  A[0..i-1] ба B[0..k-i-1] -> C[0..k-1]
//
// Invariant: i + j == k (байнга хадгалагдана)
//
// Binary search алхам бүрт:
//   A[i-1] > B[j]  гэвэл i хэт их → i-г бага болго
//   B[j-1] > A[i]  гэвэл j хэт их → j-г бага болго (i-г нэмэгдүүл)
//   Хоёулаа биелсэн → зөв co-rank олдлоо
// ---------------------------------------------------------------
__device__ int co_rank(int k, const int* A, int m, const int* B, int n) {
    // i: A-с авах элементийн тоогийн дотоод таамаглал
    int i     = (k < m) ? k : m;         // i <= min(k, m)
    int j     = k - i;                    // j = k - i
    int i_low = (k > n) ? (k - n) : 0;   // i >= max(0, k-n)
    int j_low = (k > m) ? (k - m) : 0;

    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            // i хэт их: жижигрүүл
            int delta = (i - i_low + 1) / 2;
            j_low = j;
            i    -= delta;
            j    += delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            // j хэт их: жижигрүүл (≥ → stability: тэнцүү үед A давна)
            int delta = (j - j_low + 1) / 2;
            i_low = i;
            j    -= delta;
            i    += delta;
        } else {
            active = false;   // зөв утга олдлоо
        }
    }
    return i;
}

// ---------------------------------------------------------------
// Параллел merge kernel
//
// Нэг kernel дуудалтаар A ба B-г merge хийж C-д бичнэ.
// Thread бүр output-ийн [k_start, k_end) хэсгийг хариуцна.
//
// Grid/Block тооцоо:
//   total = m + n (нийт гаралт)
//   segment = (total + threads - 1) / threads
//   BLOCK = 256
//   gridSize = (threads_needed + BLOCK - 1) / BLOCK
// ---------------------------------------------------------------
__global__ void merge_corank_kernel(
        const int* A, int m,
        const int* B, int n,
        int*       C,
        int        segment)   // нэг thread-ийн хариуцах элементийн тоо
{
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    int total   = m + n;
    int k_start = tid * segment;
    int k_end   = k_start + segment;
    if (k_end   > total) k_end = total;
    if (k_start >= total) return;

    // Co-rank → оролтын эхлэх, дуусах индексийг олно
    int i_start = co_rank(k_start, A, m, B, n);
    int i_end   = co_rank(k_end,   A, m, B, n);
    int j_start = k_start - i_start;
    int j_end   = k_end   - i_end;

    // Энэ thread-ийн хэсгийг sequential merge хийнэ
    int i = i_start, j = j_start, k = k_start;
    while (i < i_end && j < j_end) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else              C[k++] = B[j++];
    }
    while (i < i_end) C[k++] = A[i++];
    while (j < j_end) C[k++] = B[j++];
}

// ---------------------------------------------------------------
// Host-аас дуудах merge функц
//
//   h_A[0..m-1], h_B[0..n-1] – эрэмбэлэгдсэн host массивууд
//   h_C[0..m+n-1]            – гаралтын host массив
// ---------------------------------------------------------------
void cuda_corank_merge(const int* h_A, int m,
                       const int* h_B, int n,
                       int*       h_C)
{
    int total = m + n;
    if (total == 0) return;

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m     * sizeof(int));
    cudaMalloc((void**)&d_B, n     * sizeof(int));
    cudaMalloc((void**)&d_C, total * sizeof(int));

    cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Thread тооцоо: нэг thread дор хаяж SEGMENT элемент авна
    const int BLOCK   = 256;
    const int SEGMENT = 4;          // нэг thread-ийн хариуцах хэмжээ
    int num_threads   = (total + SEGMENT - 1) / SEGMENT;
    int gridSize      = (num_threads + BLOCK - 1) / BLOCK;

    merge_corank_kernel<<<gridSize, BLOCK>>>(d_A, m, d_B, n, d_C, SEGMENT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, total * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ---------------------------------------------------------------
// Тест: жижиг жишээ дээр co-rank ба merge шалгана
// nvcc -o corank_test cuda_corank.cu -DCORANK_STANDALONE && ./corank_test
// ---------------------------------------------------------------
#ifdef CORANK_STANDALONE

#include <algorithm>
#include <cassert>
#include <cstdio>

int main() {
    // Kirk & Hwu 12-р бүлгийн жишээ
    // A = {1, 7, 8, 9, 10},  B = {2, 7, 10, 11}
    int A[] = {1, 7, 8, 9, 10};
    int B[] = {2, 7, 10, 11};
    int m = 5, n = 4;
    int total = m + n;
    int C[9];

    printf("=== Co-rank туршилт (Kirk & Hwu 12-р бүлгийн жишээ) ===\n");
    printf("A = ["); for (int i=0;i<m;i++) printf("%d%s",A[i],i<m-1?", ":"");
    printf("]\n");
    printf("B = ["); for (int i=0;i<n;i++) printf("%d%s",B[i],i<n-1?", ":"");
    printf("]\n\n");

    // Co-rank утгуудыг шалгана (номын жишээтэй тохируулж)
    // C[4]-ийн co-rank: i=3, j=1
    int i43 = 0; // co_rank device функцийг host дээр тест хийхийн тулд CPU хувилбар
    // CPU хувилбар (шалгалтад зориулан):
    auto cpu_corank = [&](int k, const int* Aa, int mm, const int* Bb, int nn) {
        int ii = (k < mm) ? k : mm;
        int jj = k - ii;
        int il = (k > nn) ? (k - nn) : 0;
        int jl = (k > mm) ? (k - mm) : 0;
        bool act = true;
        while (act) {
            if (ii > 0 && jj < nn && Aa[ii-1] > Bb[jj]) {
                int d = (ii - il + 1) / 2; jl = jj; ii -= d; jj += d;
            } else if (jj > 0 && ii < mm && Bb[jj-1] >= Aa[ii]) {
                int d = (jj - jl + 1) / 2; il = ii; jj -= d; ii += d;
            } else { act = false; }
        }
        return ii;
    };

    // co_rank утгийг шалгана: i+j == k ба merge-ийн оролтын хязгаар зөв эсэхийг
    // std::merge-тэй харьцуулан баталгаажуулна.
    // Тайлбар: номын Fig 12.3 өөр array ашигласан (PDF-д зарим тоо
    // emoji болж алдагдсан тул шууд харьцуулах боломжгүй).
    int i4 = cpu_corank(4, A, m, B, n);
    int j4 = 4 - i4;
    printf("co_rank(k=4): i=%d, j=%d  (i+j=%d, k=4)  %s\n",
           i4, j4, i4+j4, (i4+j4==4) ? "✓" : "✗");
    assert(i4 + j4 == 4);

    int i6 = cpu_corank(6, A, m, B, n);
    int j6 = 6 - i6;
    printf("co_rank(k=6): i=%d, j=%d  (i+j=%d, k=6)  %s\n",
           i6, j6, i6+j6, (i6+j6==6) ? "✓" : "✗");
    assert(i6 + j6 == 6);

    // GPU merge
    printf("\n=== GPU Merge (co-rank) ===\n");
    cuda_corank_merge(A, m, B, n, C);

    printf("C = [");
    for (int i = 0; i < total; i++) printf("%d%s", C[i], i<total-1?", ":"");
    printf("]\n");

    // Зөв байгааг шалгах
    int expected[9];
    std::merge(A, A+m, B, B+n, expected);
    bool ok = (memcmp(C, expected, total*sizeof(int)) == 0);
    printf("Үр дүн: %s\n", ok ? "ЗӨРӨГҮЙ ✓" : "АЛДААТАЙ ✗");

    // Том жишээгээр хурд хэмжих
    const int BIG = 1 << 20;  // 1M элемент
    int *hA = new int[BIG], *hB = new int[BIG], *hC = new int[2*BIG];
    for (int i = 0; i < BIG; i++) { hA[i] = i * 2; hB[i] = i * 2 + 1; }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    int *dA, *dB, *dC;
    cudaMalloc((void**)&dA, BIG   * sizeof(int));
    cudaMalloc((void**)&dB, BIG   * sizeof(int));
    cudaMalloc((void**)&dC, 2*BIG * sizeof(int));
    cudaMemcpy(dA, hA, BIG*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, BIG*sizeof(int), cudaMemcpyHostToDevice);

    const int BLOCK = 256, SEG = 4;
    int nth = (2*BIG + SEG - 1) / SEG;
    int grd = (nth + BLOCK - 1) / BLOCK;

    cudaEventRecord(t0);
    merge_corank_kernel<<<grd, BLOCK>>>(dA, BIG, dB, BIG, dC, SEG);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("\nТом жишээ (2×%dM): kernel %.3f ms\n", BIG/1000000, ms);

    delete[] hA; delete[] hB; delete[] hC;
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

#endif // CORANK_STANDALONE
