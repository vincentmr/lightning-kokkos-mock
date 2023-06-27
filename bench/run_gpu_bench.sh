export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=$PWD/../build/src:$PATH
if [ -f timings.dat ]; then
    cp timings.dat copy_of_timings.dat
fi
rm -f timings.dat
echo `which 04_Exercise`
PROC=7
# for PROC in 7; do
# for type in ref mdr tv1 tv2 ttr; do
for type in ref; do
    rm -f timings.dat
    echo "NPROC, SIZE, NQBITS, TARGET, NREPEAT, TIME, BANDWIDTH" >> timings.dat

    export OMP_NUM_THREADS=$((2**PROC))
    printf -v ONT %04d $OMP_NUM_THREADS
    for NBIT in `seq 16 27`; do
        MIDTARG=$((NBIT / 2))
        MAXTARG=$((NBIT - 1))
        NREPS=100
        if [ $NBIT -gt 25 ]; then 
            NREPS=25; 
        fi 
        for TARG in $(seq $MAXTARG $MAXTARG); do
            TMP="$ONT, "
            echo -n "$TMP" >> timings.dat
            lightning_kokkos -S $NBIT -T $TARG -nrepeat 1 --type $type
            lightning_kokkos -S $NBIT -T $TARG -nrepeat $NREPS --type $type >> timings.dat
        done
    done

    cp timings.dat timings_$type.dat
done