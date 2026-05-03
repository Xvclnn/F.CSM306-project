#include "tasksys.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace {

inline void check_cuda(cudaError_t error, const char* action) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(error));
    }
}

inline int normalize_threads_per_block(int requested_threads) {
    int threads = requested_threads > 0 ? requested_threads : 256;
    if (threads < 32) {
        threads = 32;
    }

    const int warp_remainder = threads % 32;
    if (warp_remainder != 0) {
        threads += 32 - warp_remainder;
    }

    if (threads > 1024) {
        threads = 1024;
    }

    return threads;
}

__device__ __forceinline__ int min_int(int a, int b) {
    return a < b ? a : b;
}

__device__ int co_rank_cuda(int k,
                            const float* left_array,
                            int left_array_size,
                            const float* right_array,
                            int right_array_size) {
    int low = (k > right_array_size) ? (k - right_array_size) : 0;
    int high = (k < left_array_size) ? k : left_array_size;

    while (low <= high) {
        int i = (low + high) / 2;
        int j = k - i;

        if (i > 0 && j < right_array_size && left_array[i - 1] > right_array[j]) {
            high = i - 1;
        } else if (j > 0 && i < left_array_size && right_array[j - 1] > left_array[i]) {
            low = i + 1;
        } else {
            return i;
        }
    }

    return low;
}

__device__ void merge_corank_serial_cuda(const float* left_array,
                                         int left_array_size,
                                         const float* right_array,
                                         int right_array_size,
                                         float* output_array) {
    int i = 0;
    int l = 0;
    int r = 0;

    while (l < left_array_size && r < right_array_size) {
        if (left_array[l] <= right_array[r]) {
            output_array[i] = left_array[l];
            ++i;
            ++l;
        } else {
            output_array[i] = right_array[r];
            ++i;
            ++r;
        }
    }

    while (l < left_array_size) {
        output_array[i] = left_array[l];
        ++i;
        ++l;
    }

    while (r < right_array_size) {
        output_array[i] = right_array[r];
        ++i;
        ++r;
    }
}

__global__ void merge_pass_kernel(const float* input_array,
                                  float* output_array,
                                  int array_size,
                                  int width) {
    const int pair_index = blockIdx.x;
    const int left_start = pair_index * (2 * width);

    if (left_start >= array_size) {
        return;
    }

    const int left_array_size = min_int(width, array_size - left_start);
    const int right_start = left_start + left_array_size;
    const int right_array_size = (right_start < array_size)
        ? min_int(width, array_size - right_start)
        : 0;
    const int merged_size = left_array_size + right_array_size;

    const float* left_array = input_array + left_start;
    const float* right_array = input_array + right_start;

    const int thread_index = threadIdx.x;
    const int thread_count = blockDim.x;
    const int k_start = (thread_index * merged_size) / thread_count;
    const int k_end = ((thread_index + 1) * merged_size) / thread_count;

    const int left_start_rank = co_rank_cuda(
        k_start, left_array, left_array_size, right_array, right_array_size);
    const int right_start_rank = k_start - left_start_rank;

    const int left_end_rank = co_rank_cuda(
        k_end, left_array, left_array_size, right_array, right_array_size);
    const int right_end_rank = k_end - left_end_rank;

    merge_corank_serial_cuda(left_array + left_start_rank,
                             left_end_rank - left_start_rank,
                             right_array + right_start_rank,
                             right_end_rank - right_start_rank,
                             output_array + left_start + k_start);
}

}  // namespace

void TaskSystemCUDA::run_sort(int num_threads, float* array, int array_size) {
    if (array_size <= 1) {
        return;
    }

    float* device_input = nullptr;
    float* device_temp = nullptr;

    auto cleanup = [&]() {
        if (device_input != nullptr) {
            cudaFree(device_input);
            device_input = nullptr;
        }
        if (device_temp != nullptr) {
            cudaFree(device_temp);
            device_temp = nullptr;
        }
    };

    try {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA device detected.");
        }

        const int threads_per_block = normalize_threads_per_block(num_threads);
        const size_t bytes = static_cast<size_t>(array_size) * sizeof(float);

        check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc(device_input)");
        check_cuda(cudaMalloc(&device_temp, bytes), "cudaMalloc(device_temp)");
        check_cuda(cudaMemcpy(device_input, array, bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy host to device");

        float* current_input = device_input;
        float* current_output = device_temp;

        for (int width = 1; width < array_size; width *= 2) {
            const int merges_in_pass = (array_size + (2 * width) - 1) / (2 * width);
            merge_pass_kernel<<<merges_in_pass, threads_per_block>>>(
                current_input, current_output, array_size, width);
            check_cuda(cudaGetLastError(), "merge_pass_kernel launch");
            std::swap(current_input, current_output);
        }

        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        check_cuda(cudaMemcpy(array, current_input, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy device to host");
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
}
