for a in 10 15 20
do
  printf "%8d\n" $a
  for k in 1 2 4 8 16 32
  do
    printf "\t%8d\n" $k
    ofile=out_s_${a}_${k}.csv
    for s in 1 2 4 8 16 32
    do
      printf "\t\t%8d\n" $s
      srun ./bench 10000 $s $k $a 128 0 >> tmp
    done
    echo 'streams,throughput' >> $ofile
    awk '{print $2, $7}' tmp >> $ofile
    rm tmp
  done
done
