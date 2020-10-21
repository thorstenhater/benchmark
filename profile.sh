nsc=/opt/nvidia/nsight-compute/2020.1.0/nv-nsight-cu-cli
nss=/opt/nvidia/nsight-systems/2020.2.5/bin/nsys
kase=bench_graph_split_update

make -B CASE=$kase

srun $nss profile --trace=cuda,nvtx --output=$kase --force-overwrite=true --sample=cpu bench 2 16 32 25 128 2
#$run $nsc         --export=arbor-solver-with-shmem-nsc --force-overwrite -s 1000 -c 100 bench 32 16 32 25 128 10
