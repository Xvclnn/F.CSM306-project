#pragma once
#include "itasksys.h"

// ---------------------------------------------------------------
// 1. Дараалсан (Sequential)
// ---------------------------------------------------------------
class TaskSystemSerial : public ITaskSystem {
public:
    Node* sort(Node* head) override;
};

// ---------------------------------------------------------------
// 2. std::thread – CPU multi-threading
//    hardware_concurrency()-г ашиглан thread тоог тодорхойлно.
//    Жижиг жагсаалтад thread-ийн overhead sequential-аас ихэсдэг
//    тул threshold-с доош sequential рүү ордог.
// ---------------------------------------------------------------
class TaskSystemThread : public ITaskSystem {
    int maxDepth;   // log2(hardware_concurrency)
public:
    TaskSystemThread();
    Node* sort(Node* head) override;
};

// ---------------------------------------------------------------
// 3. OpenMP – compiler directive-р параллел
//    #pragma omp task ашиглан recursive parallelism хийнэ.
//    Lec5: task creation overhead-с их ажил байхад л ашигтай.
// ---------------------------------------------------------------
class TaskSystemOpenMP : public ITaskSystem {
public:
    Node* sort(Node* head) override;
};

// ---------------------------------------------------------------
// 4. CUDA – GPU дээрх хувилбар
//    Linked list-ийг array болгон GPU руу илгээж, bottom-up
//    merge sort хийгээд буцааж list-д бичнэ.
//    Тусдаа хэмжилт хадгална: H->D, kernel, D->H (Lec6 шаардлага)
// ---------------------------------------------------------------
#ifdef USE_CUDA
class TaskSystemCUDA : public ITaskSystem {
public:
    double h2d_ms    = 0.0;  // host -> device дамжуулалтын хугацаа
    double kernel_ms = 0.0;  // GPU kernel-ийн цэвэр ажиллах хугацаа
    double d2h_ms    = 0.0;  // device -> host дамжуулалтын хугацаа

    Node* sort(Node* head) override;
};
#endif
