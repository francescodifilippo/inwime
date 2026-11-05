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
