#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <thread>

using timer = std::chrono::high_resolution_clock;
using usecs = std::chrono::microseconds;

static double delta_t(const timer::time_point t0, const timer::time_point t1) {
  return std::chrono::duration_cast<usecs>(t1 - t0).count();
}

struct benchmark_parameters {
  size_t threads, blocks;
  size_t repetitions, epochs;
  size_t array_size, array_size_per_kernel;
  size_t slots, kernels_per_slot;
};

template<typename T>
T* offset_parameters(T* ptr, size_t step, size_t idx) {
  return ptr + idx*step;
}

template<typename T>
T offset_parameter(T val, size_t, size_t) {
  return val;
}
