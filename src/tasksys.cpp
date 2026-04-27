#include "tasksys.h"

void reset_array(float* base_array, float* sorting_array, int array_size){
    memcpy(sorting_array, base_array, array_size * sizeof(float));
}

