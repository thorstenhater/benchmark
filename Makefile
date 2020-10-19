bench: streams.cu util_cuda.h kernels.cu main.cu util.hpp
	nvcc -c -O3 -std=c++17 kernels.cu
	nvcc -c -O3 -std=c++17 main.cu
	nvcc -O3 kernels.o main.o -o $@

clean:
	rm -rf bench a.out
