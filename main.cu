#include <chrono> 
#include <iostream>
#include <string>
#include <thread>
#include <vector>


#include "graphs.hcu"

// read command line arguments
int read_arg(int argc, char** argv, int index, int default_value) {
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

int main(int argc, char** argv) {
    benchmark_parameters parameters;
    parameters.epochs           = read_arg(argc, argv, 1, 10);
    parameters.slots            = read_arg(argc, argv, 2, 1);
    parameters.kernels_per_slot = read_arg(argc, argv, 3, 4);
    parameters.array_size       = 1 << read_arg(argc, argv, 4, 20);
    parameters.threads          = read_arg(argc, argv, 5, 128);
    parameters.use_threading    = read_arg(argc, argv, 6, 0);

    auto total_kernels = parameters.slots*parameters.kernels_per_slot;
    auto elements_per_kernel = parameters.array_size/total_kernels;
    parameters.blocks = (elements_per_kernel + parameters.threads - 1)/parameters.threads;

    std::cout << "array_size          = " << parameters.array_size  << std::endl;
    std::cout << "epochs              = " << parameters.epochs << std::endl;
    std::cout << "slots               = " << parameters.slots << std::endl;
    std::cout << "kernels_per_slots   = " << parameters.kernels_per_slot << std::endl;
    std::cout << "block_dim           = " << parameters.threads << std::endl;
    std::cout << "threading           = " << (parameters.use_threading ? std::to_string(parameters.slots) + " threads" : "1 thread") << std::endl;

    double* xh = malloc_host<double>(parameters.array_size);
    double* yh = malloc_host<double>(parameters.array_size);

    double* xd = malloc_device<double>(parameters.array_size);
    double* yd = malloc_device<double>(parameters.array_size);

    std::fill(xh, xh + parameters.array_size, 2.0);
    std::fill(yh, yh + parameters.array_size, 1.0);

    copy_to_device<double>(xh, xd, parameters.array_size);
    copy_to_device<double>(yh, yd, parameters.array_size);

    start_gpu_prof();
    auto result = bench_graph(parameters, empty);
    stop_gpu_prof();

    for (const auto t: result) {
      std::cout << t << '\n';
    }

    std::free(xh);
    std::free(yh);
    free_device(xd);
    free_device(yd);
}
