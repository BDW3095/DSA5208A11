import pandas as pd
import matplotlib.pyplot as plt
from SGDfit import *
import os
dataDirectory = 'processedData/'
# dataDirectory = '/mnt/d/test/25_09/'
PLOT_DIR = None

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank   = comm.Get_rank()

Xtrain = collectiveRead(dataDirectory + 'X_train.csv', comm, nprocs, rank)
ytrain = collectiveRead(dataDirectory + 'y_train.csv', comm, nprocs, rank)
Xtest = collectiveRead(dataDirectory + 'X_test.csv', comm, nprocs, rank)
ytest = collectiveRead(dataDirectory + 'y_test.csv', comm, nprocs, rank)

#-----helper for graphing-------#
def _plot_dir(dirname: str = "training_plots") -> str:
    """Create/return folder next to this file for saving plots."""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, dirname) #take the directory that contains this file
    os.makedirs(path, exist_ok=True) #Create the folder (and parents) if missing; do nothing if it already exists.
    return path

def _elbow_index(x: np.ndarray, y: np.ndarray):
    """
    Find 'elbow' as the point with max perpendicular distance to the straight
    line joining the first and last logged points. Returns index or None.
    """
    # if len(x) < 3:
    #     return None
    # x0, y0 = x[0],  y[0]
    # x1, y1 = x[-1], y[-1]
    # vx, vy = (x1 - x0), (y1 - y0)
    # denom = np.hypot(vx, vy)
    # if denom == 0:
    #     return None
    # # Perpendicular distance from each (x_i, y_i) to the line (x0,y0)-(x1,y1)
    # dist = np.abs(vy * (x - x0) - vx * (y - y0)) / denom
    # # Exclude endpoints; take argmax over interior
    # idx = int(np.argmax(dist[1:-1]) + 1)
    # return idx
    n = len(x)
    if n < 3:
        return None

    xfit = np.log1p(x)  # compress large iteration gaps; keeps order
    best_i, best_sse = None, np.inf

    # try each interior i as the breakpoint
    for i in range(1, n - 1):
        # segment A: [0..i]
        p1 = np.polyfit(xfit[:i+1], y[:i+1], 1)
        y1 = np.polyval(p1, xfit[:i+1])
        sse1 = np.sum((y[:i+1] - y1) ** 2)

        # segment B: [i..n-1]
        p2 = np.polyfit(xfit[i:], y[i:], 1)
        y2 = np.polyval(p2, xfit[i:])
        sse2 = np.sum((y[i:] - y2) ** 2)

        sse = sse1 + sse2
        if sse < best_sse:
            best_sse = sse
            best_i = i

    return best_i

def _iterations(history_len: int, cycle: int):
    """Return [0, cycle, 2*cycle, ...] matching trainLossHistory length."""
    return np.arange(history_len, dtype=int) * int(cycle)

def _save_training_curve(trainLossHistory: np.ndarray, cycle: int,
                         actv_f: str, batch_size: int, width: int,
                         out_dir: str):
    # if out_dir is None:
    #     out_dir = _plot_dir()   # fallback if not provided
    if trainLossHistory is None or len(trainLossHistory) == 0:
        return

    x = _iterations(len(trainLossHistory), cycle)
    plt.figure()
    plt.plot(x, trainLossHistory, linewidth=1.8) # just the line no dot

    # # 2) biggest decrease location: argmin of first differences
    # if len(trainLossHistory) >= 2: #find the iteration that the MSE decreasing the most
    #     diffs = np.diff(trainLossHistory)  # MSE[t] - MSE[t-1]
    #     idx = int(np.argmin(diffs))  # most negative change
    #     x_drop = x[idx + 1]  # iteration at the end of that drop
    #     y_drop = float(trainLossHistory[idx + 1])
    #     plt.axvline(x_drop, color='red', linestyle='--', linewidth=1.5)
    #     plt.scatter([x_drop], [y_drop], color='red', s=32, zorder=3)

    # (2) elbow by max distance to the chord (first-to-last line)
    idx = _elbow_index(x, trainLossHistory)
    if idx is not None:
        x_knee = x[idx]
        y_knee = float(trainLossHistory[idx])
        plt.axvline(x_knee, color='red', linestyle='--', linewidth=1.5)
        plt.scatter([x_knee], [y_knee], color='red', s=32, zorder=3)

     ## make the fluctuation more easily to notice
    tail = trainLossHistory[1:] if len(trainLossHistory) > 1 else trainLossHistory
    if len(tail) > 0:
        upper = float(np.percentile(tail, 99))
        upper = max(upper, float(tail[-1]))
        plt.ylim(bottom=0.0, top=upper * 1.1)


    plt.xlabel("Iteration")
    plt.ylabel("Training MSE")
    plt.title(f"MSE vs Iteration ({actv_f}, bs={batch_size}, w={width}, c={cycle})")
    plt.grid(True) # draws grid lines at the major ticks on both x and y.

    fname = f"mse_{actv_f}_bs{batch_size}_w{width}_c{cycle}.png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()





if rank == 0:
    df_result = pd.DataFrame(
        # columns=["activation_function", "width", "learning_rate_0", "learning_rate_1", "batch_size", "seed",
        #          "threshold", "cycle", "train_loss_history", "test_loss"])
        columns = ["activation_function", "width",  "batch_size",  "training RMSE","Testing RMSE"])
    PLOT_DIR = _plot_dir()


for atv_f in ["tanh"]:#,"relu" "softplus", "tanh", "sigmoid"]:
    for batch_size in [2048]:#, 128,256, 512, 1024, 2048]:
        for width in [128]:#, 16, 32, 64, 128]:
            # for n_process in [1,2,4,8]:
            params = [atv_f, width, 0.001, 0.001, batch_size, 5208, 0.001, 100]
            trainLossHistory, testLoss = trainAndReport(
                comm, nprocs, rank, Xtrain, ytrain, Xtest, ytest, params
            )
            if rank == 0:
                df_result.loc
                train_rmse_last = float(np.sqrt(trainLossHistory[-1]))
                test_rmse = float(np.sqrt(testLoss))
                df_result.loc[len(df_result)] = [
                    params[0],
                    params[1],
                    params[4],
                    train_rmse_last,
                    test_rmse
                ]
                # PASS the global directory here:
                _save_training_curve(
                    trainLossHistory, params[7], params[0], params[4], params[1],
                    out_dir=PLOT_DIR
                )
if rank == 0:
    # df_result.to_csv("the_result.csv",index=False)
    # print("Saved CSV: the_result.csv")
    out_path = os.path.abspath("the_result.csv")
    df_result.to_csv(out_path, index=False)
    print(f"Saved CSV to: {out_path}")



