#pragma once
#include "itasksys.h"

void reset_array(float* base_array, float* sorting_array, int array_size);
void merge_sort(float* array, int array_size);
void merge(float* leftArray, float* rightArray, float* array, int array_size);

class TaskSystemSerial : public TaskSystem {
public:
    void run_sort(int num_threads, float* array, int array_size) override;
};

class TaskSystemThread : public TaskSystem {
public:
    void run_sort(int num_threads, float* array, int array_size) override;
};

class TaskSystemOpenMP : public TaskSystem {
public:
    void run_sort(int num_threads, float* array, int array_size) override;
};

#ifdef USE_CUDA
class TaskSystemCUDA : public TaskSystem {
public:
    void run_sort(int num_threads, float* array, int array_size) override;
};
#endif
