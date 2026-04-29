#include "tasksys.h"
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Лекц 5 – Program Optimization:
 *   Параллел давуу тал гарахын тулд хуваагдаж буй ажил нь
 *   thread/task үүсгэх зардлыг нөхөх хэмжээний их байх ёстой.
 *   Туршилтаар тодорхойлсон хязгаар: дор хаяж ~4096 элемент
 *   байвал параллелчлах нь sequential-аас хурдан болно.
 *
 * Лекц 3 – Latency & Bandwidth:
 *   Linked list нь санах ойд тасалдмал байрлалтай (pointer chase),
 *   тиймээс cache locality муу. Сайн locality-г хадгалахын тулд
 *   OpenMP хувилбарт array руу хуулж, тэнд bottom-up sort хийнэ.
 */

// Жижиг жагсаалтад parallel thread үүсгэх нь overhead-ыг нэмнэ.
// Дор хаяж энэ тооны элемент байхад л thread spawn хийнэ.
static const int THREAD_THRESHOLD = 4096;

// ================================================================
// Нийтлэг туслах функцүүд
// ================================================================

static int listLen(Node* head) {
    int n = 0;
    for (Node* p = head; p != nullptr; p = p->next) n++;
    return n;
}

// Iterative merge – stack overflow гарахаас сэргийлнэ
static Node* listMerge(Node* a, Node* b) {
    Node dummy(0);
    Node* tail = &dummy;
    while (a != nullptr && b != nullptr) {
        if (a->val <= b->val) { tail->next = a; a = a->next; }
        else                  { tail->next = b; b = b->next; }
        tail = tail->next;
    }
    tail->next = (a != nullptr) ? a : b;
    return dummy.next;
}

// Floyd-ын slow/fast pointer-р хоёр хагас болгоно
static void listSplit(Node* head, Node** left, Node** right) {
    Node* slow = head;
    Node* fast = head->next;
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }
    *left  = head;
    *right = slow->next;
    slow->next = nullptr;
}

// ================================================================
// 1. Дараалсан (Sequential) merge sort
// ================================================================

static Node* seqSort(Node* head) {
    if (head == nullptr || head->next == nullptr) return head;
    Node *left, *right;
    listSplit(head, &left, &right);
    left  = seqSort(left);
    right = seqSort(right);
    return listMerge(left, right);
}

Node* TaskSystemSerial::sort(Node* head) {
    return seqSort(head);
}

// ================================================================
// 2. std::thread – олон урсгалт хувилбар
//
//   Рекурс бүрт баруун хагасыг шинэ thread-д өгч, зүүнийг
//   одоогийн thread дор хийнэ. Depth хязгаар буюу жагсаалт
//   хэт жижиг болбол sequential рүү ордог.
//
//   Лекц 5: Thread creation on Linux ~10-50 μs overhead.
//   THREAD_THRESHOLD-ийн доорх жагсаалтад overhead > speedup.
// ================================================================

TaskSystemThread::TaskSystemThread() {
    unsigned hw = std::thread::hardware_concurrency();
    maxDepth = (hw > 1) ? static_cast<int>(std::log2(hw)) : 1;
}

static Node* threadSort(Node* head, int depth, int len) {
    if (head == nullptr || head->next == nullptr) return head;

    if (depth <= 0 || len < THREAD_THRESHOLD)
        return seqSort(head);

    Node *left, *right;
    listSplit(head, &left, &right);

    Node* sortedRight = nullptr;
    std::thread t([&sortedRight, right, depth, len]() {
        sortedRight = threadSort(right, depth - 1, len / 2);
    });
    left = threadSort(left, depth - 1, len / 2);
    t.join();

    return listMerge(left, sortedRight);
}

Node* TaskSystemThread::sort(Node* head) {
    int len = listLen(head);
    return threadSort(head, maxDepth, len);
}

// ================================================================
// 3. OpenMP – bottom-up iterative merge sort
//
//   Recursive task-based merge sort on linked list нь OpenMP-д
//   тохиромжгүй: task creation overhead хэт их. Иймд:
//     1) Жагсаалтын утгуудыг массив src-д хуулна
//     2) Bottom-up merge sort: width = 1, 2, 4, ..., n/2
//        Нэг level-ийн бүх merge хоорондоо хамаарахгүй тул
//        #pragma omp parallel for-р зэрэг хийнэ
//     3) Эрэмбэлэгдсэн утгуудыг жагсаалтад буцааж бичнэ
//
//   Лекц 7 – Data-Parallel Thinking:
//     "for each pair of adjacent runs → merge" гэсэн data-parallel
//     pattern нь OpenMP parallel for-тэй тохирно.
//
//   Лекц 5: Coarse-grained parallelism (том ажлын хэсэг) нь
//     fine-grained (олон жижиг task) -ээс overhead бага.
// ================================================================

Node* TaskSystemOpenMP::sort(Node* head) {
    if (head == nullptr || head->next == nullptr) return head;

    int n = listLen(head);

    // Жагсаалтын утгуудыг массивт хуулна (cache-friendly болно)
    std::vector<int> src(n), dst(n);
    {
        Node* p = head;
        for (int i = 0; i < n; i++, p = p->next) src[i] = p->val;
    }

    // Bottom-up merge sort: level бүрт parallel for ажиллуулна
    for (int width = 1; width < n; width *= 2) {

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int left = 0; left < n; left += 2 * width) {
            int mid   = std::min(left + width, n);
            int right = std::min(left + 2 * width, n);
            // std::merge нь [left,mid) ба [mid,right)-г нэгтгэж
            // dst[left..right)-д бичнэ
            std::merge(src.begin() + left, src.begin() + mid,
                       src.begin() + mid, src.begin() + right,
                       dst.begin() + left);
        }
        // src ба dst-г солих (шинэ level-д src нь эрэмбэлэгдсэн байна)
        std::swap(src, dst);
    }

    // Эрэмбэлэгдсэн утгуудыг жагсаалтын node-уудад бичнэ
    {
        Node* p = head;
        for (int i = 0; i < n; i++, p = p->next) p->val = src[i];
    }

    return head;
}
