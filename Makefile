NVCC      := nvcc
ARCH      ?= sm_86
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -Iinclude -lineinfo --extended-lambda

SRCS      := $(wildcard main/*.cu main/*.cpp src/common/*.cu src/common/*.cpp src/graph/*.cu src/graph/*.cpp src/SSSP/*.cu src/SSSP/*.cpp)
TARGET    := sssp_runner

# Dataset & Batch Files
GRAPH     := cage12.txt
BATCHES   := cage12_batches.txt
SRC_NODE  := 1
PREFIX    := cage12
ARGS      := $(GRAPH) $(BATCHES) --source $(SRC_NODE) --one-indexed --out-prefix $(PREFIX)

# Nsight Systems Profiling & Log Output
NSYS          := nsys
IMPORTER      := /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
NSYS_OUT      := result/profile_cage12
NSYS_LOG      := result/nsys_kernel_summary.log
NSYS_CSV      := result/nsys_kernel_summary.csv

.PHONY: all run nsys clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(ARGS)

nsys: $(TARGET)
	ncu --metrics gpu__time_duration.sum --csv --log-file result/kernel_timings.csv ./$(TARGET) $(ARGS)
	@cat result/kernel_timings.csv

clean:
	rm -f $(TARGET) $(NSYS_OUT).nsys-rep $(NSYS_OUT).qdstrm $(NSYS_OUT).sqlite $(NSYS_LOG) $(NSYS_CSV)