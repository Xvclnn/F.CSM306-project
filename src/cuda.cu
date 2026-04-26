// cuda.cu – CUDA implementation of Merge Sort on Singly Linked List
// Strategy:
//   1. Walk the list and copy values into a host array  (O(n))
//   2. Transfer array to device                          (H->D)
//   3. Bottom-up merge sort entirely on the GPU
//   4. Transfer sorted array back to host               (D->H)
//   5. Write values back to the existing list nodes     (O(n))
//
// This keeps the linked-list interface intact while exploiting
// GPU parallelism for the sort itself.

#include "tasksys.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// -----------------------------------------------------------
// Device helper: merge two adjacent sorted runs in-place
//   arr[left .. mid-1]  and  arr[mid .. right-1]
//   result written to temp, then copied back to arr
// -----------------------------------------------------------
__device__ void deviceMerge(int* arr, int* temp, int left, int mid, int right) {
    int i = left, j = mid, k = left;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else                  temp[k++] = arr[j++];
    }
    while (i < mid)   temp[k++] = arr[i++];
    while (j < right) temp[k++] = arr[j++];

    // Write merged result back to arr
    for (int x = left; x < right; x++) arr[x] = temp[x];
}

// -----------------------------------------------------------
// Kernel: each thread handles one pair of adjacent runs
//   width – current sub-array half-width (1, 2, 4, 8, …)
// -----------------------------------------------------------
__global__ void mergeSortKernel(int* arr, int* temp, int n, int width) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int left = tid * 2 * width;

    if (left >= n) return;

    int mid   = min(left + width,         n);
    int right = min(left + 2 * width,     n);

    if (mid < right)   // there is actually a right run to merge
        deviceMerge(arr, temp, left, mid, right);
}

// -----------------------------------------------------------
// TaskSystemCUDA::sort
// -----------------------------------------------------------
Node* TaskSystemCUDA::sort(Node* head) {
    // Count nodes
    int n = 0;
    for (Node* p = head; p; p = p->next) n++;
    if (n <= 1) return head;

    // Linearise linked list -> host array
    int* h_arr = new int[n];
    {
        Node* p = head;
        for (int i = 0; i < n; i++, p = p->next) h_arr[i] = p->val;
    }

    // Allocate device buffers
    int* d_arr  = nullptr;
    int* d_temp = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_arr),  n * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&d_temp), n * sizeof(int));

    // Copy data host -> device
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Bottom-up merge sort on the GPU
    const int BLOCK = 256;
    for (int width = 1; width < n; width *= 2) {
        // Number of merge pairs needed
        int pairs    = (n + 2 * width - 1) / (2 * width);
        int gridSize = (pairs + BLOCK - 1) / BLOCK;
        mergeSortKernel<<<gridSize, BLOCK>>>(d_arr, d_temp, n, width);
        cudaDeviceSynchronize();
    }

    // Copy sorted data device -> host
    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_temp);

    // Write sorted values back into the existing list nodes
    {
        Node* p = head;
        for (int i = 0; i < n; i++, p = p->next) p->val = h_arr[i];
    }

    delete[] h_arr;
    return head;
}
