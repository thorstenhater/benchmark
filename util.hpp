#pragma once

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

inline double delta_t(const timer::time_point t0, const timer::time_point t1) {
  return (t1 - t0).count();
}

struct benchmark_parameters {
  size_t threads, blocks;
  size_t repetitions, epochs;
  size_t array_size, array_size_per_kernel;
  size_t slots, kernels_per_slot;
};

template<typename T>
inline T* offset_parameter(T* ptr, size_t step, size_t idx) {
  return ptr + idx*step;
}

template<typename T>
inline T offset_parameter(T val, size_t, size_t) {
  return val;
}

// read command line arguments
inline int read_arg(int argc, char** argv, int index, int default_value) {
    if(argc>index) {
        try {
            auto n = std::stoi(argv[index]);
            if(n<0) {
                return default_value;
            }
            return n;
        }
        catch (std::exception e) {
            std::cout << "error : invalid argument \'" << argv[index]
                      << "\', expected a positive integer." << std::endl;
            exit(1);
        }
    }
    return default_value;
}
