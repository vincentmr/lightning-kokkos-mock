export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=$PWD/../build/src:$PATH
if [ -f timings.dat ]; then
    cp timings.dat copy_of_timings.dat
fi
rm -f timings.dat
echo `which lightning_kokkos`
echo "NPROC, SIZE, NQBITS, TARGET, NREPEAT, TIME, BANDWIDTH" >> timings.dat
for PROC in `seq 7 -1 0`; do
    export OMP_NUM_THREADS=$((2**PROC))
    printf -v ONT %04d $OMP_NUM_THREADS
    for NBIT in `seq 11 30`; do
        MAXTARG=$((NBIT - 1))
        for TARG in $(seq 0 $MAXTARG); do
            TMP="$ONT, "
            echo -n "$TMP" >> timings.dat
            if [ $OMP_NUM_THREADS -ge 32 ];then
                lightning_kokkos -S $NBIT -T $TARG -nrepeat 100 >> timings.dat
            else
                lightning_kokkos -S $NBIT -T $TARG -nrepeat  10 >> timings.dat
            fi
        done
    done
done