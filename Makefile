CXX = g++-13
CXXFLAGS = -O3 -std=c++17 -march=native -fopenmp -Wall -Wextra -Wpedantic
LDLIBS = -lcublas -lcudart

all: build/bf16_stream

build/bf16_stream: src/benchmark.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

test: all
	python3 tests/smoke.py

build/bf16_stream_profile: src/benchmark.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -DBENCH_NVTX $< -o $@ $(LDLIBS) -lnvToolsExt

profile: build/bf16_stream_profile

build/bf16_stream_trace: src/benchmark.cpp src/activity_trace.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -DBENCH_CUPTI $< -o $@ $(LDLIBS) -lcupti

trace: build/bf16_stream_trace

.PHONY: all test profile trace

# Standalone CPU adapter; no CUDA, BLAS, Python headers or system installation.
build/libinwime_cpu.so: src/cpu_kernel.c
	mkdir -p build
	$(CC) -O3 -std=c11 -fPIC -shared -fno-fast-math -ffp-contract=off -Wall -Wextra -Wpedantic $< -o $@

cpu-kernel: build/libinwime_cpu.so

.PHONY: cpu-kernel

# Operators of the Percorso additivo (#41); separate from the pinned kernel above.
build/libinwime_ops.so: src/inwime_ops.c src/inwime_ops.h
	mkdir -p build
	$(CC) -O3 -std=c11 -fPIC -shared -fno-fast-math -ffp-contract=off -fno-builtin -Wall -Wextra -Wpedantic $< -o $@ -lm

ops-kernel: build/libinwime_ops.so

.PHONY: ops-kernel
