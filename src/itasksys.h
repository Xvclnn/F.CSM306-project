#pragma once

class TaskSystem {
public:
    virtual void run_sort(int num_threads, float* array, int array_size) = 0;
    virtual ~TaskSystem() = default;
};
