#include <chrono>
#include "tasksys.h"
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cmath>

const int NUM_THREADS = 20;
const int TOTAL_RUN = 10;

// Жагсаалтыг бүрэн дараалал орсон эсэхийг шалгаж, хэрэв ороогүй байвал error заана.
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

    // csv файлын толгой мөрийг хэвлэх нь
    fprintf(fp, "method,input_size,num_threads,run_id,execution_time_ms,data_transfer_time,data_transferred_bytes,total_operations,achievable_performance\n");
        
    int array_sizes[3] = {10000, 100000, 1000000}; // тооцоолол хийх жагсаалтуудын хэмжээ

    for(int run_id = 1; run_id<=TOTAL_RUN; run_id++){
        for(int array_size : array_sizes){ 

            int total_operations = 2*array_size*ceil(log2(array_size));

            float* sorting_array = (float*)malloc(array_size * sizeof(float));
            float* base_array = (float*)malloc(array_size * sizeof(float));

            for(int i = 0; i<array_size; i++) {  // Array-н элементүүдэд санамсаргүй утга оноох нь
                random = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                result = 1.0f + random * (1000.0f - 1.0f);
                base_array[i] = result;
            }
            // Сая үүсгэсэн үндсэн жагсаалтыг одоо дараалалд оруулах гэж буй жагсаалтад хуулах
            memcpy(sorting_array, base_array, array_size * sizeof(float));

            //Serial-n test hiine
            {
                TaskSystemSerial sys;
                reset_array(base_array, sorting_array, array_size);
                auto start = std::chrono::high_resolution_clock::now();
                sys.run_sort(1, sorting_array, array_size);
                auto end = std::chrono::high_resolution_clock::now();
                checker(sorting_array, array_size);
                double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

                double achievable_performance = total_operations / (execution_time / 1000.0);

                printf("Serial, execution time: %.5f ms\n", execution_time);
                printf("Serial, achievable performance: %lf op/s\n", achievable_performance);
                // method,input_size,num_threads,run_id,execution_time,data_transfer_time,data_transferred_bytes,total_operations,achievable_performance
                fprintf(fp, "serial,%d,%d,%d,%lf,0,0,%d, %lf\n", array_size, 1, run_id, execution_time, total_operations, achievable_performance);

            }

            // std::thread test hiine
            {
                TaskSystemThread sys;
                reset_array(base_array, sorting_array, array_size);
                auto start = std::chrono::high_resolution_clock::now();
                sys.run_sort(NUM_THREADS, sorting_array, array_size);
                auto end = std::chrono::high_resolution_clock::now();
                checker(sorting_array, array_size);
                double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

                double achievable_performance = total_operations / (execution_time / 1000.0);

                printf("std::threads, execution time: %.5f ms\n", execution_time);
                printf("std:threads, achievable performance: %lf op/s\n", achievable_performance);
                fprintf(fp, "threads,%d,%d,%d,%lf,0,0,%d, %lf\n", array_size, NUM_THREADS, run_id, execution_time, total_operations, achievable_performance);

            }

            // openMP test hiine
            {
                TaskSystemOpenMP sys;
                reset_array(base_array, sorting_array, array_size);
                auto start = std::chrono::high_resolution_clock::now();
                sys.run_sort(NUM_THREADS, sorting_array, array_size);
                auto end = std::chrono::high_resolution_clock::now();
                checker(sorting_array, array_size);
                double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

                double achievable_performance = total_operations / (execution_time / 1000.0);

                printf("OpenMP, execution time: %.5f ms\n", execution_time);
                printf("OpenMP, achievable performance: %lf op/s\n", achievable_performance);
                fprintf(fp, "openmp,%d,%d,%d,%lf,0,0,%d, %lf\n", array_size, NUM_THREADS, run_id, execution_time, total_operations, achievable_performance);
            }

#ifdef USE_CUDA
            // CUDA (GPU) test hiine
            // Анхаар: USE_CUDA макро тодорхойлогдсон үед л хөрвүүлэгдэнэ.
            //   nvcc -DUSE_CUDA ... src/main.cpp src/tasksys.cpp src/cuda.cu
            {
                TaskSystemCUDA sys;
                reset_array(base_array, sorting_array, array_size);
                auto start = std::chrono::high_resolution_clock::now();
                sys.run_sort(NUM_THREADS, sorting_array, array_size);
                auto end = std::chrono::high_resolution_clock::now();
                checker(sorting_array, array_size);
                double execution_time = std::chrono::duration<double, std::milli>(end - start).count();

                double achievable_performance = total_operations / (execution_time / 1000.0);

                printf("CUDA,   execution time: %.5f ms (transfer: %.5f ms, %lld bytes)\n",
                       execution_time, sys.data_transfer_time_ms, sys.data_transferred_bytes);
                printf("CUDA,   achievable performance: %lf op/s\n", achievable_performance);
                // method,input_size,num_threads,run_id,execution_time_ms,
                // data_transfer_time,data_transferred_bytes,total_operations,achievable_performance
                fprintf(fp, "cuda,%d,%d,%d,%lf,%lf,%lld,%d, %lf\n",
                        array_size, 256 /* CUDA block size */, run_id, execution_time,
                        sys.data_transfer_time_ms, sys.data_transferred_bytes,
                        total_operations, achievable_performance);
            }
#endif
            free(sorting_array);
            free(base_array);
        }
    }
    fclose(fp);
}