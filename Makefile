bench: simple.hcu streams.hcu graphs.hcu util_cuda.h kernels.hcu kernels.cu main.cu util.hpp
	nvcc -DCASE=${CASE} -c -O3 -std=c++17 kernels.cu
	nvcc -DCASE=${CASE} -c -O3 -std=c++17 main.cu
	nvcc -DCASE=${CASE} -O3 kernels.o main.o -o $@

.PHONY: clean
clean:
	rm -rf bench a.out *.o
