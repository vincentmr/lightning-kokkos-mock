export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=$PWD:$PATH
if [ -f timings.dat ]; then
    cp timings.dat copy_of_timings.dat
fi
rm -f timings.dat
echo `which ex1`
PROC=7

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
        ex1 -S $NBIT -T $TARG -nrepeat 1 --type $type
        ex1 -S $NBIT -T $TARG -nrepeat $NREPS --type $type >> timings.dat
    done
done
