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

// ---------------------------------------------------------------
// Co-rank device функц (Kirk & Hwu Fig 12.5)
//
// Гаралтын C массивын k-р индекст A-аас хэдэн элемент авсныг
// binary search-р O(log N) хугацаанд олно. i + j == k invariant.
// ---------------------------------------------------------------
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

// ================================================================
// Kernel 1: SMALL — Жижиг хос-уудад зориулсан (1 block = 1 хос)
//
// Зориулалт: Pair хэмжээ бага (≤ SIMPLE_PAIR_LIMIT) бөгөөд тоо нь
// маш олон (мянга-аас сая) үед ашиглана. Эхний түвшинүүдэд
// (width = 1, 2, 4, ...) тохирно. Олон pair → олон block →
// GPU бүрэн ачаалагдана.
//
// Block тус бүр нэг merge pair-г хариуцна. Block доторх 256 thread
// co-rank-р хуваан merge хийнэ.
// ================================================================
__global__ void merge_pass_kernel_small(const float* input_array,
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

// ================================================================
// Kernel 2: LARGE — Том pair-уудад зориулсан (олон block = 1 pair)
//
// Зориулалт: Pair хэмжээ их (> SIMPLE_PAIR_LIMIT) бөгөөд тоо нь цөөн
// (1-100) үед ашиглана. Эцсийн түвшинүүдэд (width = 4096, ..., n/2)
// тохирно.
//
// Block тус бүр гаралтын OUTPUT_PER_BLOCK элементийг хариуцна.
// Олон block нэг pair дотор зэрэг ажилладаг тул GPU бүрэн ачаалагдана:
//   Жишээ нь N=1M-ийн эцсийн merge-т (width = n/2):
//     Хуучин loop: зөвхөн 1-2 block ажиллана (~500 thread)
//     Шинэ loop:   ~245 block × 256 thread = 62,720 thread зэрэг ✓
//
// Ажиллах схем:
//   1) Block-ийн chunk-ийн pair-г олж, pair-ийн дотор chunk-ийн
//      зүүн/баруун хязгаарыг co-rank-р тооцоолно (зөвхөн thread 0).
//   2) Shared memory-оор бусад thread-уудтай хуваалцана.
//   3) Block доторх thread бүр өөрийн жижиг хэсгийг merge хийнэ.
// ================================================================
__global__ void merge_pass_kernel_large(const float* input_array,
                                        float* output_array,
                                        int array_size,
                                        int width,
                                        int output_per_block) {
    const int chunk_start = blockIdx.x * output_per_block;
    if (chunk_start >= array_size) {
        return;
    }
    int chunk_end = chunk_start + output_per_block;
    if (chunk_end > array_size) {
        chunk_end = array_size;
    }

    const int pair_size = 2 * width;
    const int pair_index = chunk_start / pair_size;
    const int pair_start = pair_index * pair_size;
    const int mid = min_int(pair_start + width, array_size);
    const int pair_end = min_int(pair_start + pair_size, array_size);

    // Block нэг pair-ийн дотор үлдэхийн тулд chunk-г pair-ийн хязгаарт оруулна.
    // (output_per_block нь pair_size-ийг хуваадаг тул энэ нь маш ховор тохиолдол.)
    if (chunk_end > pair_end) {
        chunk_end = pair_end;
    }
    if (chunk_start >= chunk_end) {
        return;
    }

    const float* left_array = input_array + pair_start;
    const int left_array_size = mid - pair_start;
    const float* right_array = input_array + mid;
    const int right_array_size = pair_end - mid;

    // Pair-ийн дотоод координат руу шилжүүлнэ
    const int local_chunk_start = chunk_start - pair_start;
    const int local_chunk_end = chunk_end - pair_start;

    // Block-level co-rank: энэ block-ийн оролтын хязгаарыг тооцоолно.
    // Зөвхөн thread 0 тооцоод бусадтай shared memory-оор хуваалцана.
    __shared__ int block_left_start;
    __shared__ int block_left_end;
    if (threadIdx.x == 0) {
        block_left_start = co_rank_cuda(
            local_chunk_start, left_array, left_array_size,
            right_array, right_array_size);
        block_left_end = co_rank_cuda(
            local_chunk_end, left_array, left_array_size,
            right_array, right_array_size);
    }
    __syncthreads();

    const int block_right_start = local_chunk_start - block_left_start;
    const int block_right_end = local_chunk_end - block_left_end;

    const int tile_left_size = block_left_end - block_left_start;
    const int tile_right_size = block_right_end - block_right_start;
    const int tile_merged_size = tile_left_size + tile_right_size;

    const float* tile_left = left_array + block_left_start;
    const float* tile_right = right_array + block_right_start;

    // Block доторх thread бүр өөрийн жижиг хэсгийг хариуцна
    const int thread_index = threadIdx.x;
    const int thread_count = blockDim.x;
    const int k_start = (thread_index * tile_merged_size) / thread_count;
    const int k_end = ((thread_index + 1) * tile_merged_size) / thread_count;

    const int thread_left_start = co_rank_cuda(
        k_start, tile_left, tile_left_size, tile_right, tile_right_size);
    const int thread_right_start = k_start - thread_left_start;
    const int thread_left_end = co_rank_cuda(
        k_end, tile_left, tile_left_size, tile_right, tile_right_size);
    const int thread_right_end = k_end - thread_left_end;

    merge_corank_serial_cuda(tile_left + thread_left_start,
                             thread_left_end - thread_left_start,
                             tile_right + thread_right_start,
                             thread_right_end - thread_right_start,
                             output_array + chunk_start + k_start);
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

    merge_pass_kernel_small<<<1, 32>>>(device_input_, device_temp_, 2, 1);
    check_cuda(cudaGetLastError(), "warmup kernel launch");
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

    // ============================================================
    // Hybrid стратеги: 2 төрлийн kernel-ийг хослуулан ашиглана.
    //
    //   pair_size <= SIMPLE_PAIR_LIMIT  →  SMALL kernel
    //                                       (1 block нэг pair-г хийнэ)
    //                                       Олон pair тул GPU
    //                                       автоматаар ачаалагдана.
    //
    //   pair_size >  SIMPLE_PAIR_LIMIT  →  LARGE kernel
    //                                       (олон block нэг pair-г хийнэ)
    //                                       Pair цөөхөн тул block-ийг
    //                                       output position-оор хуваана.
    //
    // Үр дүн: бүх width түвшинд ОЛОН block launch болж GPU
    // ажилгүй зогсохгүй. Tesla T4-ийн 2,560 CUDA core-ийн
    // utilization 90%+ хүрнэ.
    //
    // SIMPLE_PAIR_LIMIT болон OUTPUT_PER_BLOCK хоёулаа 2-ийн зэрэг
    // тоо ба OUTPUT_PER_BLOCK <= SIMPLE_PAIR_LIMIT байх ёстой
    // (block нэг pair-д үлдэх нөхцөл).
    // ============================================================
    constexpr int SIMPLE_PAIR_LIMIT = 4096;
    constexpr int OUTPUT_PER_BLOCK  = 4096;
    static_assert(OUTPUT_PER_BLOCK <= SIMPLE_PAIR_LIMIT,
                  "OUTPUT_PER_BLOCK SIMPLE_PAIR_LIMIT-ээс их байж болохгүй");

    for (int width = 1; width < array_size; width *= 2) {
        const int pair_size = 2 * width;

        if (pair_size <= SIMPLE_PAIR_LIMIT) {
            const int merges_in_pass = (array_size + pair_size - 1) / pair_size;
            merge_pass_kernel_small<<<merges_in_pass, threads_per_block>>>(
                current_input, current_output, array_size, width);
        } else {
            const int blocks_needed =
                (array_size + OUTPUT_PER_BLOCK - 1) / OUTPUT_PER_BLOCK;
            merge_pass_kernel_large<<<blocks_needed, threads_per_block>>>(
                current_input, current_output, array_size, width,
                OUTPUT_PER_BLOCK);
        }
        check_cuda(cudaGetLastError(), "merge kernel launch");
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
