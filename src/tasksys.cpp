#include "tasksys.h"
#include <vector>
#include <cstring>

void reset_array(float* base_array, float* sorting_array, int array_size){
    memcpy(sorting_array, base_array, array_size * sizeof(float));
}

void merge_sort(float* array) {

}