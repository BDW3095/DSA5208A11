# run_and_plot.py
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv


# import your existing functions (unchanged)
from SGDfit_jacky1 import trainAndReport, collectiveRead




# ----------------- internal cache so we only load once -----------------
_DATA_CACHE = None

def _get_data(comm, nprocs, rank, data_dir="processedData/"):
    """
    Load train/test splits once using your collectiveRead.
    Returns cached arrays on subsequent calls.
    """
    global _DATA_CACHE
    if _DATA_CACHE is None:
        Xtrain = collectiveRead(data_dir + "X_train.csv", comm, nprocs, rank)
        ytrain = collectiveRead(data_dir + "y_train.csv", comm, nprocs, rank)
        Xtest  = collectiveRead(data_dir + "X_test.csv",  comm, nprocs, rank)
        ytest  = collectiveRead(data_dir + "y_test.csv",  comm, nprocs, rank)
        _DATA_CACHE = (Xtrain, ytrain, Xtest, ytest)
    return _DATA_CACHE


# ----------------- small helper -----------------
def build_iterations(loss_hist, cycle):
    """
    Create [0, cycle, 2*cycle, ...] with the same length as loss_hist.
    This is computed HERE (not in SGDfit_jacky1.py), as requested.
    """
    cycle = int(cycle)
    return np.arange(len(loss_hist), dtype=int) * cycle

# ================= big-curve helpers =================

def ema(y, alpha=0.1):
    """Exponential moving average; alpha in (0,1]."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out

def downsample_curve(x, y, max_points=1000, mode="mean"):
    """
    Reduce to at most max_points via fixed-size bins.
    mode: 'mean' | 'median' | 'minmax'
    Returns:
      - mean/median: (xd, yd)
      - minmax:      (xd, ymean, ymin, ymax)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(y)
    if n <= max_points:
        if mode == "minmax":
            return x, y, y, y
        return x, y

    win = int(np.ceil(n / max_points))
    pad = (-n) % win
    if pad:
        x = np.pad(x, (0, pad), mode="edge")
        y = np.pad(y, (0, pad), mode="edge")
    X = x.reshape(-1, win)
    Y = y.reshape(-1, win)

    if mode == "median":
        xd = np.median(X, axis=1)
        yd = np.median(Y, axis=1)
        return xd, yd
    if mode == "minmax":
        xd = X.mean(axis=1)
        ymean = Y.mean(axis=1)
        ymin = Y.min(axis=1)
        ymax = Y.max(axis=1)
        return xd, ymean, ymin, ymax
    # mean
    xd = X.mean(axis=1)
    yd = Y.mean(axis=1)
    return xd, yd


# ----------------- main API -----------------
def run_experiment(params, data_dir="processedData/"):
    """
    Call your trainAndReport and return a dict of everything we need.

    params layout (same as your trainer):
      [actv, width, lrate0, lrate1, nrandrows, seed, threshold, cycle]
    """
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    nprocs = comm.Get_size()

    Xtrain, ytrain, Xtest, ytest = _get_data(comm, nprocs, rank, data_dir)

    # call your existing function (unchanged)
    trainLossHistory, testLoss, trainTime = trainAndReport(
        comm, nprocs, rank, Xtrain, ytrain, Xtest, ytest, params
    )

    cycle = int(params[7])
    iterationList = build_iterations(trainLossHistory, cycle)

    # package
    return {
        "loss": np.asarray(trainLossHistory, dtype=float),
        "iters": iterationList,
        "test_mse": (float(testLoss) if testLoss is not None else None),
        "elapsed_sec": float(trainTime),
        "params": params,
        "nprocs": nprocs,
    }



def plot_training_curve(result, label=None, outfile=None, show=False,
                        max_points=1000, smooth_alpha=None,
                        use_logy=False, y_clip_pct=None, band_minmax=False):
    """
    Plot MSE vs iteration using lists built HERE.
    Handles very long curves via smoothing & downsampling.
      - max_points: cap plotted points (binning)
      - smooth_alpha: EMA smoothing (e.g., 0.1). None to disable.
      - use_logy: True for semilogy
      - y_clip_pct: e.g., (1,99) to clip extreme spikes
      - band_minmax: fill min/max band per bin
    Only rank 0 draws/saves to avoid duplicates in MPI.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
        return None

    loss = result["loss"]
    iters = result["iters"]
    if len(loss) == 0:
        print("Nothing to plot: empty loss history (non-root rank or training failed).")
        return None

    actv, width, l0, l1, nrandrows, seed, thr, cycle = result["params"]
    nprocs = result["nprocs"]
    M_per_rank = int(int(nrandrows) / nprocs)

    # optional smoothing (apply on full-res)
    y = loss
    if smooth_alpha is not None and 0 < smooth_alpha <= 1:
        y = ema(y, alpha=smooth_alpha)

    # downsample
    if band_minmax:
        xd, ymean, ymin, ymax = downsample_curve(iters, y, max_points=max_points, mode="minmax")
    else:
        xd, yd = downsample_curve(iters, y, max_points=max_points, mode="mean")

    # decide markers: keep only for small series
    use_markers = (len(loss) <= 300)

    plt.figure()
    ax = plt.gca()
    if use_logy:
        ax.set_yscale("log")

    if band_minmax:
        plt.fill_between(xd, ymin, ymax, alpha=0.15, linewidth=0)
        plt.plot(xd, ymean, linewidth=1.6,
                 marker="o" if use_markers else None, markevery=20 if use_markers else None)
    else:
        plt.plot(xd, yd, linewidth=1.6,
                 marker="o" if use_markers else None, markevery=20 if use_markers else None)

    title = f"MSE vs Iterations ({actv}, n={width}, M/rank={M_per_rank}, cyc={cycle}, p={nprocs})"
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.4)

    if y_clip_pct is not None and len(loss) > 2:
        lo, hi = np.percentile(y, [y_clip_pct[0], y_clip_pct[1]])
        if lo < hi:
            plt.ylim(lo, hi)

    plt.tight_layout()

    if outfile is None:
        outfile = f"training_curve_{actv}_n{width}_M{M_per_rank}_cyc{cycle}_p{nprocs}.png"
    plt.savefig(outfile, dpi=200)
    if show:
        plt.show()
    plt.close()
    return outfile

def save_curve_csv(result, outdir="results"):
    """
    Save (iteration, mse) pairs to CSV (rank 0 only).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank != 0:
        return None

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    actv, width, l0, l1, nrandrows, seed, thr, cycle = result["params"]
    nprocs = result["nprocs"]
    M_per_rank = int(int(nrandrows) / nprocs)

    fname = outdir / f"curve_{actv}_n{width}_M{M_per_rank}_cyc{cycle}_p{nprocs}.csv"
    with fname.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "mse"])
        for i, m in zip(result["iters"], result["loss"]):
            w.writerow([int(i), float(m)])
    return str(fname)

# -------------- convenience: run multiple and overlay ---------------
def run_and_plot(params_list, data_dir="processedData/", show=False):
    """
    Run several configs and overlay curves in one figure (rank 0 only).
    Returns the list of result dicts (one per params).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    results = []
    for p in params_list:
        results.append(run_experiment(p, data_dir=data_dir))

    if rank == 0:
        plt.figure()
        for res in results:
            actv, width, l0, l1, nrandrows, seed, thr, cycle = res["params"]
            nprocs = res["nprocs"]
            M_per_rank = int(int(nrandrows) / nprocs)
            label = f"{actv}, n={width}, M/rank={M_per_rank}, cyc={cycle}"
            iters, loss = res["iters"], res["loss"] #change for big data
            xd, yd = downsample_curve(iters, loss, max_points=800, mode="mean")  # NEW: downsample overlay series
            plt.plot(xd, yd, linewidth=1.3, label=label) #change for big data
        plt.xlabel("Iteration"); plt.ylabel("MSE"); plt.title("Training MSE vs Iterations")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig("training_curves_overlay.png", dpi=150)
        if show: plt.show()
        plt.close()

    return results

if __name__ == "__main__":
    # params: [actv, width, lrate0, lrate1, nrandrows, seed, threshold, cycle]
    params = ["softplus", 64, 0.001, 0.001, 1024, 5208, 1e-3, 100]
    res = run_experiment(params) #when data is small just use res and ignore png
    # For large histories, use the smart plot:
    png = plot_training_curve(
        res,
        max_points=1000,  # target ~1000 points on the figure
        smooth_alpha=0.10,  # EMA smoothing; set None to disable
        use_logy=False,  # True if loss spans orders of magnitude
        y_clip_pct=None,  # e.g., (1, 99) to clip outliers
        band_minmax=True  # draw min/max band per bin for variability
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        if res["loss"].size > 0:
            train_mse_last = float(res["loss"][-1])
            train_rmse = float(np.sqrt(train_mse_last))
        else:
            train_mse_last = None
            train_rmse = None

        # test MSE comes from trainAndReport (may be None on non-root)
        test_mse = res["test_mse"]
        test_rmse = float(np.sqrt(test_mse)) if test_mse is not None else None


        print(f"Test MSE: {res['test_mse']}, time(s): {res['elapsed_sec']:.2f}")
        print("Saved plot:", png)
        print(f"Train last MSE: {train_mse_last}")
        print(f"Train last RMSE: {train_rmse}")
        print(f"Test RMSE: {test_rmse}")
        #csv_path = save_curve_csv(res)

        #print("Saved csv :", csv_path)

