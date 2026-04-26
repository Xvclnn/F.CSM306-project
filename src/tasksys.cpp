#include "tasksys.h"
#include <thread>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// =============================================================
// Shared helpers (used by all CPU variants)
// =============================================================

// Iterative merge – avoids deep recursion for large lists
static Node* listMerge(Node* a, Node* b) {
    Node dummy(0);
    Node* tail = &dummy;
    while (a && b) {
        if (a->val <= b->val) { tail->next = a; a = a->next; }
        else                  { tail->next = b; b = b->next; }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}

// Split list into two halves using slow/fast pointer
static void listSplit(Node* head, Node** front, Node** back) {
    Node* slow = head;
    Node* fast = head->next;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    *front = head;
    *back  = slow->next;
    slow->next = nullptr;
}

// =============================================================
// 1. Sequential
// =============================================================

static Node* seqSort(Node* head) {
    if (!head || !head->next) return head;
    Node *front, *back;
    listSplit(head, &front, &back);
    front = seqSort(front);
    back  = seqSort(back);
    return listMerge(front, back);
}

Node* TaskSystemSerial::sort(Node* head) {
    return seqSort(head);
}

// =============================================================
// 2. std::thread
//    Spawns a sibling thread for the right half at each level
//    until depth reaches 0, then falls back to sequential.
// =============================================================

TaskSystemThread::TaskSystemThread() {
    unsigned hw = std::thread::hardware_concurrency();
    // depth d => up to 2^d parallel sub-tasks
    threadDepth = (hw > 1) ? static_cast<int>(std::log2(hw)) : 1;
}

static Node* threadedSort(Node* head, int depth) {
    if (!head || !head->next) return head;

    Node *front, *back;
    listSplit(head, &front, &back);

    if (depth > 0) {
        // Sort right half in a new thread
        Node* sortedBack = nullptr;
        std::thread t([&]() {
            sortedBack = threadedSort(back, depth - 1);
        });
        front = threadedSort(front, depth - 1);
        t.join();
        back = sortedBack;
    } else {
        // Depth exhausted – fall back to sequential
        front = seqSort(front);
        back  = seqSort(back);
    }
    return listMerge(front, back);
}

Node* TaskSystemThread::sort(Node* head) {
    return threadedSort(head, threadDepth);
}

// =============================================================
// 3. OpenMP – task-based parallelism
//    Each recursive call spawns two omp tasks until depth 0.
// =============================================================

TaskSystemOpenMP::TaskSystemOpenMP() {
#ifdef _OPENMP
    int threads = omp_get_max_threads();
    taskDepth = (threads > 1) ? static_cast<int>(std::ceil(std::log2(threads))) : 1;
#else
    taskDepth = 1;
#endif
}

static Node* ompSort(Node* head, int depth) {
    if (!head || !head->next) return head;

    Node *front, *back;
    listSplit(head, &front, &back);

    if (depth > 0) {
#ifdef _OPENMP
        #pragma omp task shared(front)
        front = ompSort(front, depth - 1);

        #pragma omp task shared(back)
        back = ompSort(back, depth - 1);

        #pragma omp taskwait
#else
        front = seqSort(front);
        back  = seqSort(back);
#endif
    } else {
        front = seqSort(front);
        back  = seqSort(back);
    }
    return listMerge(front, back);
}

Node* TaskSystemOpenMP::sort(Node* head) {
    Node* result = head;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single nowait
        result = ompSort(head, taskDepth);
    }
#else
    result = ompSort(head, taskDepth);
#endif
    return result;
}
