#include "tasksys.h"
#include <vector>
#include <cstring>
#include <cstdlib>

//Дараах функц нь анх main.cpp дээр үүсгэсэн жагсаалтыг дахин онооно. Ингэснээр нэг л 
//утгатай адилхан жагсаалтыг sort-лох болно.
void reset_array(float* base_array, float* sorting_array, int array_size){
    memcpy(sorting_array, base_array, array_size * sizeof(float));
}

//
static void merge(float* leftArray, float* rightArray, float* array, int array_size){
    int leftArray_size = array_size/2;
    int rightArray_size = array_size - leftArray_size;
    int i=0, l=0, r=0;

    while(l < leftArray_size && r < rightArray_size) {
        if(leftArray[l] < rightArray[r]) {
            array[i] = leftArray[l];
            l++;
            i++;
        }
        else{
            array[i] = rightArray[r];
            i++;
            r++; 
        }
    }
    while(l < leftArray_size) {
        array[i] = leftArray[l];
        l++;
        i++;
    }
    while(r < rightArray_size){
        array[i] = rightArray[r];
        i++;
        r++; 
    }
}

void merge_sort(float* array, int array_size) {
    if(array_size <= 1) {
        return; //Хэрэв жагсаалтын хэмжээ нэг болон түүнээс бага болсон тохиолдолд recursive байдлыг зогсоох
    }
    int middle = array_size/2;
    float* leftArray = (float*)malloc(middle*sizeof(float));
    float* rightArray = (float*)malloc((array_size-middle)*sizeof(float));
    int i=0, j=0;

    for(;i<array_size; i++) {
        if(i < middle) {
            leftArray[i] = array[i];
        }
        else {
            rightArray[j] = array[i];
            j++;
        }
    }

    merge_sort(leftArray, middle);
    merge_sort(rightArray, array_size-middle);
    merge(leftArray, rightArray, array, array_size);
    
    free(leftArray);
    free(rightArray);
}

void TaskSystemSerial::run_sort(int NUM_THREADS, float* array, int array_size){
    // a code that utilizes merge_sort to sort out the float* array in serially
    merge_sort(array, array_size);
}

void TaskSystemThread::run_sort(int NUM_THREADS, float* array, int array_size){
    // a code that does merge sorting in parallel using std::threads
}

void TaskSystemOpenMP::run_sort(int NUM_THREADS, float* array, int array_size){
    // a code that does merge sorting in parallel using OpenMP
}