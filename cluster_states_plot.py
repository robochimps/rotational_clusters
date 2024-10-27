import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
matplotlib.rc("text", usetex=True)

if __name__ == "__main__":

    fig, ax = plt.subplots()

    with open("cluster_states_plot_pmax24.txt", "r") as fl:
        for line in fl:
            w = line.split()
            j = int(w[0])
            e = [float(ww) for ww in w[1:]]
            ax.scatter(
                [j for _ in range(len(e))],
                e,
                s=30,
                edgecolors="black",
                linewidths=0.5,
                color=cm.plasma(j / 50),
            )

    ax.set_aspect(0.08)
    ax.set_xlabel("$J$", fontsize=14)
    ax.set_ylabel("$E_{J,k,\\tau}-E_{J,k,\\tau}^{\\rm (max)},~{\\rm cm}^{-1}$", fontsize=14)
    ax.set_ylim([-600, 50])
    ax.set_xlim([0, 50])
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig('cluster_states_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
