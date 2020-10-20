#include "util.hpp"
#include "util_cuda.h"

#include "kernels.hcu"

template<typename K, typename... As>
auto bench_stream_st(const benchmark_parameters& p, K kernel, As... as) {
    std::vector<cudaStream_t> streams{n_stream, 0};
    for (auto& stream: streams) cudaStreamCreate(&stream);

    auto thread_runner = [&](unsigned stream_idx) {
        for (unsigned k = 0; k < p.kernels_per_slot; ++k) {
            unsigned kernel_idx   = k + stream_idx * p.kernels_per_slot;
            kernel<<<launch_grid_dim, block_dim, 0, streams[stream_idx]>>>(offset_parameter(as, p.array_size_per_kernel, kernel_idx)...);
        }
    };

    for (unsigned repetition = 0; repetition < p.repetitions; ++repetition) {
        auto t0 = timer::now();
        device_synch();
        for (unsigned epoch = 0; epoch < p.epochs; ++epoch) {
            for (unsigned stream_idx = 0; stream_idx < n_streams; ++stream_idx) {
                thread_runner(stream_idx);
            }
        }
        device_synch();
        auto t1 = timer::now();
        res.push_back(delta_t(t0, t1));
    }

    for (auto& stream: streams) cudaStreamDestroy(&stream);
    return res;
}

template<typename K, typename... As>
auto bench_stream_mt(const benchmark_parameters& p, K kernel, As... as) {
    std::vector<cudaStream_t> streams{n_stream, 0};
    for (auto& stream: streams) cudaStreamCreate(&stream);

    auto thread_runner = [&](unsigned stream_idx) {
        for (unsigned k = 0; k < p.kernels_per_slot; ++k) {
            unsigned kernel_idx   = k + stream_idx * p.kernels_per_slot;
            kernel<<<launch_grid_dim, block_dim, 0, streams[stream_idx]>>>(offset_parameter(as, p.array_size_per_kernel, kernel_idx)...);
        }
    };

    std::vector<double> res;
    for (unsigned repetition = 0; repetition < p.repetitions; ++repetition) {
        auto t0 = timer::now();
        device_synch();
        for (unsigned epoch = 0; epoch < p.epochs; ++epoch) {
            std::vector<std::thread> threads;
            for (unsigned stream_idx = 0; stream_idx < p.slots; ++stream_idx) {
                threads.push_back(std::thread(thread_runner, stream_idx));
            }
            // wait for all the cpu threads to have finished
            for (unsigned stream_idx = 0; stream_idx < p.slots; ++stream_idx) {
                threads[stream_idx].join();
            }
            // wait for all gpu kernels to have completed
            device_synch();

        }
        auto t1 = timer::now();
        res.push_back(delta_t(t0, t1));
    }

    for (auto& stream: streams) cudaStreamDestroy(&stream);
    return res;
}
