for a in 10 15 20
do
  printf "%8d\n" $a
  for s in 1 2 4 8 16 32
  do
    printf "\t%8d\n" $s
    ofile=out_k_${a}_${s}.csv
    for k in 1 2 4 8 16 32
    do
      printf "\t\t%8d\n" $k
      srun ./bench 10 $s $k $a 128 0 >> tmp
    done
    echo 'kernels_per_stream,throughput' >> $ofile
    awk '{print $3, $7}' tmp >> $ofile
    rm tmp
  done
done
