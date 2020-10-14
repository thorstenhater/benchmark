for a in 10 15 20
do
  printf "%8d\n" $a
  for ks in 8 16 32 64
  do
    printf "\t%8d\n" $ks
    ofile=out_ks_${a}_${ks}.csv
    for s in 1 2 4 8
    do
      let k=$ks/$s
      printf "\t\t%8d %8d\n" $s $k
      srun ./bench 10 $s $k $a 128 0 >> tmp
    done
    echo 'streams,throughput' >> $ofile
    awk '{print $2, $7}' tmp >> $ofile
    rm tmp
  done
done

