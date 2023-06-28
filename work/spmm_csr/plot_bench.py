from matplotlib import pyplot as plt
import numpy as np

filename = "timings.dat"
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


def make_plot(nproc, nqbit, ntarg, bandw, num_threads=1):
    import matplotlib.pyplot as plt

    fig = plt.figure(dpi=500)
    mask = nproc == num_threads
    nproc, nqbit, ntarg, bandw = nproc[mask], nqbit[mask], ntarg[mask], bandw[mask]
    for i in range(20, 31):
        mask = nqbit == i
        plt.plot(ntarg[mask], bandw[mask], label=f"# nq={i}")

    plt.yscale("log", base=10)
    plt.xlabel("Target qubit")
    plt.ylim((0.1, 2000))
    plt.ylabel("Bandwidth [GB/s]")
    plt.title("Bandwidth vs. target qubit")
    # plt.xticks(np.arange(min(vntot), max(vntot) + 1) + 0.4)
    # plt.xticklabels(list(range(min(vntot), max(vntot) + 1)))
    plt.legend(loc="best")
    plt.savefig(f"bandwidth_vs_target_qubit_{num_threads}_threads.png")


for n in [128]:
    make_plot(nproc, nqbit, ntarg, bandw, num_threads=n)
