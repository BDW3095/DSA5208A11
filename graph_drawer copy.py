from ExperimentResults import *
from pathlib import Path
import matplotlib.pyplot as plt


file_path=Path("/Users/wenbidong/Desktop/5208A1/trainingOutcome/lossHistory.json")
out_path="./Fina_Graph/"


with file_path.open("r") as f:
    data = json.load(f)

def get_key(d, k):
    """Works whether keys are strings ('57') or integers (57)."""
    return d.get(str(k), d.get(k))

# Extract and store in local variables
Sigmoid = get_key(data, 57)
Tanh = get_key(data, 39)
relu = get_key(data, 19)

it_fun = [Sigmoid, Tanh, relu]
labels = ["Sigmoid, bs=256, w=32, c=100","tanh, bs=256, w=32, c=100" , "relu, bs=256, w=32, c=100"]
# # Optional quick sanity check
# print(len(run57), len(run39), len(run19))
# print(run57[:3], run39[:3], run19[:3])

# print(type(run57))


# ----------------- small helper -----------------
def build_iterations(loss_hist, cycle):
    """
    Create [0, cycle, 2*cycle, ...] with the same length as loss_hist.
    This is computed HERE (not in SGDfit_jacky1.py), as requested.
    """
    cycle = int(cycle)
    return np.arange(len(loss_hist), dtype=int) * cycle


def _safe_filename(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in " .,_-=") else "_" for ch in s)



def plot_mse_vs_iteration(loss_lists, labels, cycle: int,
                                title: str ):
    plt.figure()
    i = 0
    while i < len(loss_lists):
        losses = loss_lists[i]
        lab = labels[i]
        x = build_iterations(losses, cycle)
        plt.plot(x, losses, label=lab)
        # # NEW: cap Y to hide early spikes (clear tail view)
        # y_lo = float(np.percentile(losses, 5))
        # y_hi = float(np.percentile(losses, 99))
        # plt.ylim(y_lo, y_hi)

        # --- small change to match reference scales ---
        xmax = min(x[-1], 15000)
        ymax = float(np.percentile(losses, 99))
        plt.xlim(0, xmax)
        plt.ylim(0, ymax)

        title = f"{title}"
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Training MSE")
        plt.grid(True)
        plt.legend()
        fname = f"MSE_vs_Iteration__{_safe_filename(lab)}.png"
        Path(out_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_path) / fname, dpi=150, bbox_inches="tight")
        plt.close()
        i += 1

    # for lab in labels:
    #     safe = "".join(ch if (ch.isalnum() or ch in " .,_-=") else "_" for ch in lab)
    #     plt.savefig(Path(out_path) / f"{safe}.png", dpi=150, bbox_inches="tight")



if __name__ == '__main__':
    plot_mse_vs_iteration(
        it_fun,
        labels,
        cycle=100,
        title = "MSE vs Iteration")