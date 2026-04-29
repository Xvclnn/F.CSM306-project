# F.CSM306 – Merge Sort on Singly Linked List
# Бие даалтын ажил: Sequential / std::thread / OpenMP / CUDA
#
# Хэрэглээ:
#   make          -> CPU хувилбар (default)
#   make run      -> Build + ажиллуулах + csv үүсгэх
#   make cuda     -> CUDA хувилбар (nvcc шаардлагатай)
#   make visualize -> Python графикийг зурах
#   make clean    -> Build файлуудыг устгах

UNAME := $(shell uname -s)

# ---------------------------------------------------------------
# Compiler тохиргоо: Linux эсвэл macOS автоматаар илрүүлнэ
# ---------------------------------------------------------------
ifeq ($(UNAME), Linux)
    CXX      := g++
    CXXFLAGS := -std=c++17 -O2 -fopenmp -Isrc
    LDFLAGS  := -fopenmp -lpthread -lm
    # nvcc нь Linux дээр CUDA-тай machine-д суурилдаг
    NVCC       := nvcc
    NVCCFLAGS  := -std=c++14 -O2 -Isrc -Xcompiler -fopenmp
    CUDA_LFLAGS:= -lgomp -lm
else
    # macOS: Apple clang OpenMP дэмжихгүй тул Homebrew LLVM ашиглана
    CXX      := /opt/homebrew/opt/llvm/bin/clang++
    CXXFLAGS := -std=c++17 -O2 -fopenmp \
                -I/opt/homebrew/opt/libomp/include -Isrc
    LDFLAGS  := -L/opt/homebrew/opt/libomp/lib -lomp
    NVCC       := nvcc
    NVCCFLAGS  := -std=c++14 -O2 -Isrc
    CUDA_LFLAGS:=
endif

CPU_SRCS  := src/main.cpp src/tasksys.cpp
CUDA_SRC  := src/cuda.cu
HEADERS   := src/itasksys.h src/tasksys.h

BIN       := bin
CPU_BIN   := $(BIN)/sort_benchmark
CUDA_BIN  := $(BIN)/sort_benchmark_cuda

.PHONY: all cpu cuda run run-cuda visualize clean

all: cpu

# CPU-only: Sequential + std::thread + OpenMP
cpu: $(CPU_SRCS) $(HEADERS)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(CPU_SRCS) -o $(CPU_BIN) $(LDFLAGS)
	@echo "[OK] Built $(CPU_BIN)"

# CUDA build: бүх 4 хувилбар (nvcc шаардлагатай)
cuda: $(CUDA_SRC) $(CPU_SRCS) $(HEADERS)
	@mkdir -p $(BIN)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(CUDA_SRC) $(CPU_SRCS) \
	    -o $(CUDA_BIN) $(CUDA_LFLAGS)
	@echo "[OK] Built $(CUDA_BIN)"

run: cpu
	@mkdir -p csv
	./$(CPU_BIN)

run-cuda: cuda
	@mkdir -p csv
	./$(CUDA_BIN)

visualize:
	python3 visualization/main.py

clean:
	rm -rf $(BIN)
	@echo "[OK] Cleaned."
