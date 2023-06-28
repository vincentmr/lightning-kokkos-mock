from matplotlib import pyplot as plt
import numpy as np

filename = "timings.dat"

def read_data(filename):
    with open(filename, "r") as f:
        text = f.read()

    text = text.split("\n")
    text.pop(0)

    nproc, ndata, ntarg, bandw = [], [], [], []

    for line in text:
        if not line:
            break
        c0, c1, c2, c3, c4, c5, c6 = line.split(",")
        nproc.append(int(c0))
        ndata.append(int(c1))
        ntarg.append(int(c2))
        bandw.append(float(c6))
        # if int(c0) == 4:
        #     break

    nproc, ndata, ntarg, bandw = np.array(nproc), np.array(ndata), np.array(ntarg), np.array(bandw)
    nqbit = np.log2(ndata)
    ntarg = 2**(ntarg+1)
    return nproc, nqbit, ntarg, bandw

def make_plot(filenames, num_threads=1):
    import matplotlib.pyplot as plt
    
    fig = plt.figure(dpi=500)
    for filename in filenames:
        nproc, nqbit, ntarg, bandw = read_data(filename)
        plt.plot(nqbit, bandw)

    plt.yscale("log", base=10)
    plt.xlabel("Number of qubits")
    plt.ylim((0.1, 2000))
    plt.ylabel("Bandwidth [GB/s]")
    plt.title("Bandwidth vs. number of qubits")
    # plt.xticks(np.arange(min(vntot), max(vntot) + 1) + 0.4)
    # plt.xticklabels(list(range(min(vntot), max(vntot) + 1)))
    plt.legend(["cuquantum", "zgemm_batched(target#4)"], loc="best")
    plt.savefig(f"bandwidth_vs_num_qubits_A100.png")

make_plot(["cuquantum_timings.dat", "timings.dat"], num_threads=128)
