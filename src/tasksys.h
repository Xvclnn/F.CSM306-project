#pragma once
#include "itasksys.h"

void reset_array(float* base_array, float* sorting_array, int array_size);
void merge_sort_serial(float* array, int array_size);
void merge_serial(float* leftArray, float* rightArray, float* array, int array_size);

int co_rank(int k, float* leftArray, int leftArray_size, float* rightArray, int rightArray_size0);
void merge_corank_serial(float* leftArray, int leftArray_size, float* rightArray, int rightArray_size, float* array);
void merge_parallel_corank(float* leftArray, int leftArray_size, float* rightArray, int rightArray_size, float* array, int NUM_THREADS);
void merge_sort_parallel(float* array, int array_size, int NUM_THREADS);

void merge_sort_parallel_openmp(float* array, int array_size, int NUM_THREADS);
void merge_parallel_corank_openmp(float* leftArray, int leftArray_size, float* rightArray, int rightArray_size, float* array, int NUM_THREADS);

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