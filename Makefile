# =============================================================
# F.CSM306 – Merge Sort on Singly Linked List
# Targets: cpu (default), cuda, run, visualize, clean
# =============================================================

# --- macOS / Homebrew LLVM (has OpenMP support) ---
CXX      := /opt/homebrew/opt/llvm/bin/clang++
CXXFLAGS := -std=c++17 -O2 \
            -fopenmp \
            -I/opt/homebrew/opt/libomp/include \
            -Isrc
LDFLAGS  := -L/opt/homebrew/opt/libomp/lib -lomp

# --- CUDA compiler (Linux / CUDA-enabled machine) ---
NVCC      := nvcc
NVCCFLAGS := -std=c++14 -O2 -Isrc

# --- Sources ---
CPU_SRCS  := src/main.cpp src/tasksys.cpp
CUDA_SRCS := src/cuda.cu $(CPU_SRCS)
HEADERS   := src/itasksys.h src/tasksys.h

BIN_DIR      := bin
CPU_TARGET   := $(BIN_DIR)/sort_benchmark
CUDA_TARGET  := $(BIN_DIR)/sort_benchmark_cuda

# =============================================================
.PHONY: all cpu cuda run visualize clean

all: cpu

# --- CPU-only build (Sequential + std::thread + OpenMP) ------
cpu: $(CPU_SRCS) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(CPU_SRCS) -o $(CPU_TARGET) $(LDFLAGS)
	@echo "Built: $(CPU_TARGET)"

# --- CUDA build (all 4 methods) – requires nvcc --------------
cuda: $(CUDA_SRCS) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) \
	    -Xcompiler "-fopenmp" \
	    -DUSE_CUDA \
	    $(CUDA_SRCS) \
	    -o $(CUDA_TARGET) \
	    -lgomp
	@echo "Built: $(CUDA_TARGET)"

# --- Run CPU benchmark and generate CSV ----------------------
run: cpu
	@mkdir -p csv
	./$(CPU_TARGET)

# --- Plot results from CSV -----------------------------------
visualize:
	python3 visualization/main.py

# --- Clean build artefacts -----------------------------------
clean:
	rm -rf $(BIN_DIR)
	@echo "Cleaned."
