#include "tasksys.h"
#include <vector>
#include <cstring>
#include <cstdlib>
#include <omp.h>
#include <thread>

//Дараах функц нь анх main.cpp дээр үүсгэсэн жагсаалтыг дахин онооно. Ингэснээр нэг л 
//утгатай адилхан жагсаалтыг sort-лох болно.
void reset_array(float* base_array, float* sorting_array, int array_size){
    memcpy(sorting_array, base_array, array_size * sizeof(float));
}

// Хуваагдсан баруун зүүн хоёр жагсаалтыг авч буцаад merge хйинэ.
void merge_serial(float* leftArray, float* rightArray, float* array, int array_size){
    // Баруун зүүн хоёр жагсаалтын хэмжээг тодорхойлох нь
    int leftArray_size = array_size/2;
    int rightArray_size = array_size - leftArray_size;
    int i=0, l=0, r=0; 

    // Баруун зүүн хоёр жагсаалтыг аль хэдийн тоон дараалалд орсон байна гэж үзнэ.
    // Хоёр жагсаалтын зүүн гал талаас нь эхлэх ба элемент тус бүрийг хооронд нь харьцуулж
    // аль бага хэмжээтэй байгааг нь output жагсаалтад нэмэх замаар ажиллана.
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

    //Дээрх loop зогссоны дараа үлдсэн байж болзошгүй элементыг үндсэн жагсаалтад нэгтгэх
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

// Жагсаалтын хэмжээ 1 болох хүртэл жагсаалтыг мод хэлбэрээр задлаж дараах үйлдлийг хийнэ:
// Өгөгдсөн жагсаалтыг баруун зүүн хоёр хэсэгт хувааж merge() функц уруу дамжуулна. 
void merge_sort_serial(float* array, int array_size) {
    if(array_size <= 1) {
        return; //Хэрэв жагсаалтын хэмжээ нэг болон түүнээс бага болсон тохиолдолд recursive байдлыг зогсоох
    }
    int middle = array_size/2; // Жагсаалтын хэмжээг хоёрт хуваана.

    // Баруун зүүн хоёр жагсаалтыг үүсгэнэ.
    float* leftArray = (float*)malloc(middle*sizeof(float));
    float* rightArray = (float*)malloc((array_size-middle)*sizeof(float));

    int i=0, j=0;

    // Эх жагсаалтын зүүн хэсгийг leftArray[]-д, баруун хэсгийг rightArray[]-д онооно.
    for(;i<array_size; i++){
        if(i < middle) {
            leftArray[i] = array[i];
        }
        else {
            rightArray[j] = array[i];
            j++;
        }
    }

    // Хуваагдсан жагсаалтыг нэгтгэж, recursive байдлаар дахин дуудагдана. 
    merge_sort_serial(leftArray, middle);
    merge_sort_serial(rightArray, array_size-middle);
    merge_serial(leftArray, rightArray, array, array_size); 

    // memory allocation хийсэн жагсаалтыг free() хийх нь. 
    free(leftArray);
    free(rightArray);
}

// co-rank алгоритмыг ашиглаж i утгыг олно.
// Энэхүү i утгаар дамжуулан зүүн болон баруун жагсаалтаас хэд хэдэн элемент авахыг тухайн урсгал мэдэх болно.
int co_rank(int k, float* leftArray, int leftArray_size, float* rightArray, int rightArray_size){
    int low;
    if (k > rightArray_size) {
        low = k - rightArray_size;
    }
    else {
        low = 0;
    }
    
    int high;
    if (k < leftArray_size) {
        high = k;
    }
    else {
        high = leftArray_size;
    }

    // Хэрэв голын доорх i утга дээр дараах if statement-д буй нөхцөл биелэж байна гэдэг нь
    // баруун, зүүн аль нэг жагсаалтаас хэтэрхий их утга авсан байна гэж үзээд шинээр high болон low
    // утгуудыг тооцоолно. Ингэснээр жагсаалт болгоны элементүүдийг харьцуулах ачааллаас сэргийлж ердөө хэдхэн харьцуулалт хийж 
    // хүссэн i утгыг олж чадна.
    while (low <= high){
        int i = (low+high)/2;
        int j = k-i;

        if (i>0 && j<rightArray_size && leftArray[i-1]>rightArray[j]) {
            high = i-1;
        }
        else if (j>0 && i<leftArray_size && rightArray[j-1]>leftArray[i]) {
            low = i+1;
        }
        else {
            return i;
        }
    }
    return low;
}


// Урсгал бүр дотроо serial ажиллана. Ажиллах зарчим нь энгийн merge sort алгоритмтай ялгаагүй
void merge_corank_serial(float* leftArray, int leftArray_size, float* rightArray, int rightArray_size, float* array){
    int i = 0, l = 0, r = 0;
    while (l < leftArray_size && r < rightArray_size){
        if (leftArray[l] <= rightArray[r]) {
            array[i] = leftArray[l];
            i++;
            l++;
        }
        else {
            array[i] = rightArray[r];
            i++;
            r++;
        }
    }
    while (l < leftArray_size){
        array[i] = leftArray[l];
        i++;
        l++;
    }
    while (r < rightArray_size){
        array[i] = rightArray[r];
        i++;
        r++;
    }
}
 
// Урсгалуудыг үүсгэж ажлыг нь хийлгээд буцаад join хийнэ.
void merge_parallel_corank(float* leftArray, int leftArray_size, float* rightArray, int rightArray_size, float* array, int NUM_THREADS){
    int array_size = leftArray_size + rightArray_size;
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; t++){
        threads.emplace_back([=]() {
            int k_start = t*array_size/NUM_THREADS;
            int k_end = (t+1)*array_size/NUM_THREADS;

            int l_start = co_rank(k_start, leftArray, leftArray_size, rightArray, rightArray_size);
            int r_start = k_start - l_start;

            int l_end = co_rank(k_end, leftArray, leftArray_size, rightArray, rightArray_size);
            int r_end = k_end - l_end;

            merge_corank_serial(leftArray+l_start, l_end-l_start, rightArray+r_start, r_end-r_start, array+k_start);
        });
    }
    for (auto& th : threads) th.join();
}


// Жагсаалтыг хоёр тэнцүү хэсэгт хуваана. Хэрэв жагсаалтын хэмжээ нь 10,000-аас доош болоод ирвэл
// цааш нь serial дээрх алгоритмыг ажиллуулна. 
void merge_sort_parallel(float* array, int array_size, int NUM_THREADS){
    if (array_size <= 10000 || NUM_THREADS <= 1){
        merge_sort_serial(array, array_size);
        return;
    }
    int leftArray_size  = array_size / 2;
    int rightArray_size = array_size - leftArray_size;

    float* leftArray = (float*)malloc(leftArray_size * sizeof(float));
    float* rightArray = (float*)malloc(rightArray_size * sizeof(float));

    memcpy(leftArray, array, leftArray_size*sizeof(float));
    memcpy(rightArray, array+leftArray_size, rightArray_size*sizeof(float));

    int left_threads  = NUM_THREADS / 2;
    int right_threads = NUM_THREADS - left_threads;

    // Баруун болон зүүн гэж хувааж буй энэхүү үйлдийг parallel-чилах
    std::thread left_thread(merge_sort_parallel, leftArray, leftArray_size, left_threads);
    std::thread right_thread(merge_sort_parallel, rightArray, rightArray_size, right_threads);

    left_thread.join();
    right_thread.join();

    // Дараах функцыг дуудаж параллелиар merge хийх нь
    merge_parallel_corank(leftArray, leftArray_size, rightArray, rightArray_size, array, NUM_THREADS);

    free(leftArray);
    free(rightArray);
}

// Мөн std::thread-тэй адил co-runk алгоритмыг ашиглан merge-sort алгоритмыг гүйцэтгэнэ.
void merge_parallel_corank_openmp(float* leftArray, int leftArray_size, float* rightArray, int rightArray_size, float* array, int NUM_THREADS){
    int array_size = leftArray_size + rightArray_size;

    #pragma omp taskgroup
    {
        for (int t = 0; t < NUM_THREADS; t++){
            #pragma omp task firstprivate(t, array_size, leftArray_size, rightArray_size, NUM_THREADS) shared(leftArray, rightArray, array)
            {
                int k_start = t * array_size / NUM_THREADS;
                int k_end = (t + 1) * array_size / NUM_THREADS;

                int l_start = co_rank(k_start, leftArray, leftArray_size, rightArray, rightArray_size);
                int r_start = k_start - l_start;

                int l_end = co_rank(k_end, leftArray, leftArray_size, rightArray, rightArray_size);
                int r_end = k_end - l_end;

                merge_corank_serial(leftArray + l_start, l_end - l_start, rightArray + r_start, r_end - r_start, array + k_start);
            }
        }
    }
}

void merge_sort_parallel_openmp(float* array, int array_size, int NUM_THREADS){
    if (array_size <= 10000 || NUM_THREADS <= 1){
        merge_sort_serial(array, array_size);
        return;
    }

    int leftArray_size  = array_size / 2;
    int rightArray_size = array_size - leftArray_size;

    float* leftArray = (float*)malloc(leftArray_size * sizeof(float));
    float* rightArray = (float*)malloc(rightArray_size * sizeof(float));

    memcpy(leftArray, array, leftArray_size * sizeof(float));
    memcpy(rightArray, array + leftArray_size, rightArray_size * sizeof(float));

    int left_threads  = NUM_THREADS / 2;
    int right_threads = NUM_THREADS - left_threads;

    #pragma omp taskgroup
    {
        #pragma omp task firstprivate(leftArray, leftArray_size, left_threads)
        merge_sort_parallel_openmp(leftArray, leftArray_size, left_threads);

        #pragma omp task firstprivate(rightArray, rightArray_size, right_threads)
        merge_sort_parallel_openmp(rightArray, rightArray_size, right_threads);
    }

    merge_parallel_corank_openmp(leftArray, leftArray_size, rightArray, rightArray_size, array, NUM_THREADS);

    free(leftArray);
    free(rightArray);
}

// serial sort хийнэ.
void TaskSystemSerial::run_sort(int NUM_THREADS, float* array, int array_size){
    merge_sort_serial(array, array_size);
}

//std::thread аргачлалаар merge-sort алгоритмыг co-rank ашиглан хийнэ.
void TaskSystemThread::run_sort(int NUM_THREADS, float* array, int array_size){
    merge_sort_parallel(array, array_size, NUM_THREADS);
}

//openMP аргачлалаар merge-sort алгоритмыг co-rank ашиглан хийнэ.
void TaskSystemOpenMP::run_sort(int NUM_THREADS, float* array, int array_size){
    if (NUM_THREADS <= 1){
        merge_sort_serial(array, array_size);
        return;
    }
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        #pragma omp single
        merge_sort_parallel_openmp(array, array_size, NUM_THREADS);
    }
}