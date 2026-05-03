#include "tasksys.h"

#include <cuda_runtime.h>

#include <chrono>
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

template <typename Duration>
double to_milliseconds(Duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
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

TaskSystemCUDA::~TaskSystemCUDA() {
    release_buffers();
}

void TaskSystemCUDA::release_buffers() {
    if (device_input_ != nullptr) {
        cudaFree(device_input_);
        device_input_ = nullptr;
    }
    if (device_temp_ != nullptr) {
        cudaFree(device_temp_);
        device_temp_ = nullptr;
    }
    capacity_bytes_ = 0;
}

void TaskSystemCUDA::ensure_capacity(std::size_t bytes) {
    if (bytes <= capacity_bytes_) {
        return;
    }

    release_buffers();

    check_cuda(cudaMalloc(&device_input_, bytes), "cudaMalloc(device_input_)");
    try {
        check_cuda(cudaMalloc(&device_temp_, bytes), "cudaMalloc(device_temp_)");
    } catch (...) {
        release_buffers();
        throw;
    }

    capacity_bytes_ = bytes;
}

void TaskSystemCUDA::warm_up() {
    if (warmed_up_) {
        return;
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA device detected.");
    }

    ensure_capacity(2 * sizeof(float));

    const float warmup_input[2] = {1.0f, 0.0f};
    check_cuda(cudaMemcpy(device_input_,
                          warmup_input,
                          sizeof(warmup_input),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy warmup host to device");

    merge_pass_kernel<<<1, 32>>>(device_input_, device_temp_, 2, 1);
    check_cuda(cudaGetLastError(), "warmup merge_pass_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

    warmed_up_ = true;
}

const CudaRunMetrics& TaskSystemCUDA::last_metrics() const {
    return last_metrics_;
}

void TaskSystemCUDA::run_sort(int num_threads, float* array, int array_size) {
    last_metrics_ = {};

    if (array_size <= 1) {
        return;
    }

    warm_up();

    const int threads_per_block = normalize_threads_per_block(num_threads);
    const std::size_t bytes = static_cast<std::size_t>(array_size) * sizeof(float);

    ensure_capacity(bytes);

    const auto transfer_to_device_start = std::chrono::steady_clock::now();
    check_cuda(cudaMemcpy(device_input_, array, bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy host to device");
    const auto kernel_start = std::chrono::steady_clock::now();

    float* current_input = device_input_;
    float* current_output = device_temp_;

    for (int width = 1; width < array_size; width *= 2) {
        const int merges_in_pass = (array_size + (2 * width) - 1) / (2 * width);
        merge_pass_kernel<<<merges_in_pass, threads_per_block>>>(
            current_input, current_output, array_size, width);
        check_cuda(cudaGetLastError(), "merge_pass_kernel launch");
        std::swap(current_input, current_output);
    }

    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const auto transfer_from_device_start = std::chrono::steady_clock::now();

    check_cuda(cudaMemcpy(array, current_input, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy device to host");
    const auto transfer_end = std::chrono::steady_clock::now();

    last_metrics_.kernel_time_ms = to_milliseconds(transfer_from_device_start - kernel_start);
    last_metrics_.data_transfer_time_ms =
        to_milliseconds(kernel_start - transfer_to_device_start) +
        to_milliseconds(transfer_end - transfer_from_device_start);
    last_metrics_.data_transferred_bytes = 2 * bytes;
}
