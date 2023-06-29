export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=$PWD:$PATH
if [ -f timings.dat ]; then
    cp timings.dat copy_of_timings.dat
fi
rm -f timings.dat
echo `which spmm_csr_example`
PROC=7
rm -f timings.dat
echo "NPROC, SIZE, NQBITS, TARGET, NREPEAT, TIME, BANDWIDTH" >> timings.dat

export OMP_NUM_THREADS=$((2**PROC))
# printf -v ONT %04d $OMP_NUM_THREADS
for NBIT in `seq 20 30`; do
# for NBIT in 27 28 29; do
    MIDTARG=$((NBIT / 2))
    MAXTARG=$((NBIT - 1))
    NREPS=10
    for TARG in $(seq 0 12); do
    # for TARG in $(seq 4 6); do
        TMP="$OMP_NUM_THREADS, "
        echo -n "$TMP" >> timings.dat
        ./spmm_csr_example -S $NBIT -T $TARG -nrepeat 1
        ./spmm_csr_example -S $NBIT -T $TARG -nrepeat $NREPS  >> timings.dat
    done
done
