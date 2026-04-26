#pragma once
#include "itasksys.h"

// -------------------------------------------------------
// 1. Sequential – single-threaded merge sort
// -------------------------------------------------------
class TaskSystemSerial : public ITaskSystem {
public:
    Node* sort(Node* head) override;
};

// -------------------------------------------------------
// 2. Multithreaded CPU – std::thread
//    Spawns threads up to log2(hardware_concurrency) deep
// -------------------------------------------------------
class TaskSystemThread : public ITaskSystem {
    int threadDepth;
public:
    TaskSystemThread();
    Node* sort(Node* head) override;
};

// -------------------------------------------------------
// 3. Parallel CPU – OpenMP tasks
//    Uses #pragma omp task for recursive parallelism
// -------------------------------------------------------
class TaskSystemOpenMP : public ITaskSystem {
    int taskDepth;
public:
    TaskSystemOpenMP();
    Node* sort(Node* head) override;
};

// -------------------------------------------------------
// 4. GPU – CUDA (compiled separately with nvcc)
//    Linearises list -> device sort -> reconstructs list
// -------------------------------------------------------
#ifdef USE_CUDA
class TaskSystemCUDA : public ITaskSystem {
public:
    Node* sort(Node* head) override;
};
#endif
