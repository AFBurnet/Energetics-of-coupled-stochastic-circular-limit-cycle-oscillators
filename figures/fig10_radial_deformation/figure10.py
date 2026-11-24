#!/usr/bin/env python3
"""Compute stationary radial-coupling maps and plot Fig. 10."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

import numerics

HERE = Path(__file__).resolve().parent
MAPS = ("grads", "grads_v", "covs")
STYLE = {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}",
         "font.family": "serif",
         "font.serif": ["Computer Modern Roman"], "font.size": 10,
         "axes.labelsize": 12, "axes.linewidth": .8,
         "xtick.major.size": 3.5, "ytick.major.size": 3.5,
         "xtick.major.width": .8, "ytick.major.width": .8,
         "axes.grid": False, "svg.hashsalt": "figure10"}


def calculation_grid(settings):
    return tuple(np.linspace(grid["start"], grid["stop"], grid["points"])
                 for grid in (settings["beta_grid"], settings["k_grid"]))


def load_data(root=HERE):
    data = Path(root) / "data"
    settings = json.loads((data / "settings.json").read_text())
    with np.load(data / "reference.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return settings, arrays


def compute(root, beta_points=None, k_points=None):
    """Calculate the stationary maps on the configured grid."""
    settings = json.loads((HERE / "data" / "settings.json").read_text())
    if beta_points is not None:
        settings["beta_grid"]["points"] = beta_points
    if k_points is not None:
        settings["k_grid"]["points"] = k_points
    if settings["beta_grid"]["points"] < 1 or settings["k_grid"]["points"] < 2:
        raise ValueError("Use at least one beta and two coupling values")
    betas, ks = calculation_grid(settings)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = root / "data"
    data.mkdir()
    arrays = numerics.compute(betas, ks, settings["gradient_radial_points"],
                             settings["covariance_radial_points"])
    np.savez_compressed(data / "reference.npz", **arrays)
    (data / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(data / "reference.npz")


def validate(root):
    """Replay saved rows and compare radial and coupling-grid refinements."""
    settings, saved = load_data(root)
    indices = np.unique([0, len(saved["betas"]) // 2, len(saved["betas"]) - 1])
    betas, ks = saved["betas"][indices], saved["ks"]
    nr, nc = settings["gradient_radial_points"], settings["covariance_radial_points"]
    baseline = numerics.compute(betas, ks, nr, nc)
    for field in MAPS:
        delta = np.max(np.abs(baseline[field] - saved[field][indices]))
        print(f"{field}: maximum difference over {len(indices)} rows = {delta:.3g}", flush=True)
        np.testing.assert_allclose(baseline[field], saved[field][indices], rtol=2e-10, atol=2e-13)

    rows = []
    def record(comparison, quantity, beta, k, base_setting, fine_setting, base, fine):
        rows.append(dict(comparison=comparison, quantity=quantity, beta=beta, k=k,
                         baseline_setting=base_setting, refined_setting=fine_setting,
                         baseline=base, refined=fine, absolute_difference=abs(fine-base)))

    radial = numerics.compute(betas, ks, 2*nr, 2*nc)
    for quantity in MAPS:
        points = nc if quantity == "covs" else nr
        for i, beta in enumerate(betas):
            for j, k in enumerate(ks):
                record("radial_resolution", quantity, beta, k, points, 2*points,
                       baseline[quantity][i, j], radial[quantity][i, j])

    finer_ks = np.sort(np.r_[ks, (ks[:-1]+ks[1:])/2])
    for i, beta in enumerate(betas):
        mean, variance = numerics.radial_moments(beta, finer_ks, nr)
        for quantity, values in zip(("grads", "grads_v"), (mean, variance)):
            refined = np.gradient(values/values[0], finer_ks, edge_order=1)[::2]
            for j, k in enumerate(ks):
                record("coupling_step", quantity, beta, k, ks[1]-ks[0], finer_ks[1]-finer_ks[0],
                       baseline[quantity][i, j], refined[j])

    beta = 20.
    independent = numerics.uncoupled_covariance(beta, numerics.choose_rmax(beta))
    for points in (700, 1400):
        value = float(numerics.radial_covariance(beta, [0.], points)[0])
        np.testing.assert_allclose(value, independent, rtol=0, atol=2e-7)
        record("uncoupled_factorization", "covs", beta, 0., points, "adaptive", value, independent)

    beta = .01
    upper = numerics.choose_rmax(beta)
    finite = numerics.uncoupled_covariance(beta, upper)
    infinite = numerics.uncoupled_covariance(beta, np.inf)
    record("radial_domain", "covs", beta, 0., upper, "infinity", finite, infinite)
    path = Path(root)/"data/numerics.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} numerical comparisons in {path}")


def plot(root=HERE):
    """Render every panel directly from saved numerical arrays."""
    _, arrays = load_data(root)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "deformation_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    panels = [
        ("grads", (72, 15, 262, 259), "RdBu_r",
         TwoSlopeNorm(vmin=-.006386315060384118, vcenter=0., vmax=.04613516356397618),
         [-.006, 0, .04], [r"$-0.006$", r"$0.000$", r"$0.040$"],
         r"$\displaystyle\frac{1}{\langle r\rangle_0}\frac{\mathrm{d}}{\mathrm{d}k}\langle r\rangle_k$"),
        ("grads_v", (500, 15, 263, 259), "Blues_r",
         Normalize(vmin=-1.1209909786465735, vmax=-.02081887953876906),
         [-1.1209909786465735, -.5709049290926713, -.02081887953876906],
         [r"$-1.12$", r"$-0.57$", r"$-0.02$"],
         r"$\displaystyle\frac{1}{\operatorname{Var}_0(r)}\frac{\mathrm{d}}{\mathrm{d}k}\operatorname{Var}_k(r)$"),
        ("covs", (285, 374, 265, 258), "RdBu_r",
         TwoSlopeNorm(vmin=-.02159696439489911, vcenter=0., vmax=.04),
         [-.02159696439489911, 0, .04], [r"$-0.0216$", r"$0.0000$", r"$\geq0.0400$"],
         r"$\operatorname{Cov}_k(\bar r^2,\delta^2)$"),
    ]
    betas, ks = arrays["betas"], arrays["ks"]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(351 / 72, 283 / 72))
        for letter, (key, bounds, cmap, norm, ticks, labels, label) in zip("abc", panels):
            x, y, width, height = bounds
            rect = [x / 878, (708 - y - height) / 708, width / 878, height / 708]
            axis = fig.add_axes(rect)
            image = axis.imshow(arrays[key].T, origin="lower", aspect="auto",
                                extent=[betas[0], betas[-1], ks[0], ks[-1]],
                                cmap=cmap, norm=norm, interpolation="hanning")
            axis.set(xlabel=r"$\beta$", ylabel=r"$k\;(\tau_r^{-1})$", xticks=[10, 20], yticks=[0, 1, 2, 3])
            axis.xaxis.labelpad = 1
            axis.yaxis.labelpad = 2
            axis.yaxis.label.set_rotation(90)
            axis.yaxis.label.set_fontsize(10)
            axis.text(.025, .97, f"({letter})", transform=axis.transAxes,
                      va="top", fontsize=12)
            axis.text(.54, .75, label, transform=axis.transAxes,
                      ha="center", va="center", fontsize=12)
            cax = fig.add_axes([rect[0] + rect[2] + .018, rect[1], .016, rect[3]])
            colorbar = fig.colorbar(image, cax=cax, ticks=ticks)
            colorbar.ax.set_yticklabels(labels)
        output = Path(root) / "figures"
        output.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "svg", "eps"):
            fig.savefig(output / f"figure10.{extension}", dpi=180)
        plt.close(fig)
    print(output / "figure10.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plot", "compute", "validate"))
    parser.add_argument("--outdir", type=Path, help="Run root containing data/ and figures/")
    parser.add_argument("--beta-points", type=int)
    parser.add_argument("--k-points", type=int)
    args = parser.parse_args()
    if args.command == "compute":
        if args.outdir is None:
            parser.error("compute requires --outdir NEW_FOLDER")
        compute(args.outdir, args.beta_points, args.k_points)
    elif args.command == "plot":
        plot(args.outdir or HERE)
    else:
        validate(args.outdir or HERE)


if __name__ == "__main__":
    main()
