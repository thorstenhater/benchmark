bench: streams.cu util_cuda.h
	nvcc -O3 -std=c++11 streams.cu -o bench -DXXCUDA

clean:
	rm -rf bench a.out
