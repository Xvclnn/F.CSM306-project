#include <chrono>
#include "tasksys.h"
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static void checker(float* array, int array_size){
    for(int i = 0; i<array_size-1; i++) {
        if (array_size <= 1) return;
        if(array[i] <= array[i+1]) {
            continue;
        }
        else {
            throw std::runtime_error("Array бүрэн дараалалд ороогүй байна.");
        }
    }
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    float random, result;

    FILE *fp = fopen("../csv/output.csv", "w+");
    fprintf(fp, "method,input_size,num_threads,run_id,computation_time_ms,data_transfer_time_ms,execution_time_ms,speedup,total_operations,data_transferred_bytes,achievable_performance\n");
    
    int array_sizes[3] = {10000, 100000, 1000000};
    for(int array_size : array_sizes){ 
        float* sorting_array = (float*)malloc(array_size * sizeof(float));
        float* base_array = (float*)malloc(array_size * sizeof(float));

        for(int i = 0; i<array_size; i++) {  // Array-н элементүүдэд санамсаргүй утга оноох нь
            random = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            result = 1.0f + random * (100.0f - 1.0f);
            base_array[i] = result;
        }
        memcpy(sorting_array, base_array, array_size * sizeof(float));

        //Serial-n test
        {
            TaskSystemSerial sys;
            reset_array(base_array, sorting_array, array_size);
            auto start = std::chrono::high_resolution_clock::now();
            sys.run_sort(1, sorting_array, array_size);
            auto end = std::chrono::high_resolution_clock::now();
            checker(sorting_array, array_size);
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }











        
        free(sorting_array);
        free(base_array);
    }

    fclose(fp);
}
