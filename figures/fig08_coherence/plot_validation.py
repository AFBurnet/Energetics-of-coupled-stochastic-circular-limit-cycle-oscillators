"""Plot Figure 8 coherent-cycle changes under numerical refinements."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

from figure8 import COLORS, COUPLINGS, LABELS, MARKERS, READOUTS, STYLE, read_rows

HERE = Path(__file__).resolve().parent
REFINEMENTS = (("coarse", r"$2\Delta t$"), ("fine", r"$\Delta t/2$"),
               ("burn", "Equilibration\n"+r"$\times2$"),
               ("duration", "Duration\n"+r"$\times2$"))
SIGNALS = {"oscillator_1": ("x", "#444444", r"$z_1$"),
           "collective": ("o", COLORS["cartesian"], r"$Z$")}


def point(ax, x, row, marker, color, scale=1.):
    if row is None or row.get("plot_included") == "False":
        return
    value, low, high = [scale*float(row[k]) for k in
                        ("endpoint_difference", "ci_low", "ci_high")]
    if not np.isfinite([value, low, high]).all():
        return
    if low <= value <= high:
        ax.errorbar(x, value, yerr=[[value-low], [high-value]], fmt=marker,
                    color=color, mfc="white", ms=4, lw=1, capsize=2)
    else:
        ax.errorbar(x, (low+high)/2, yerr=(high-low)/2, fmt="none", color=color, lw=1, capsize=2)
        ax.plot(x, value, marker, color=color, mfc="white", ms=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE, help="Directory containing validation tables")
    parser.add_argument("--output", type=Path, help="Validation SVG path")
    args = parser.parse_args()
    output = args.output or args.outdir/"validation.svg"
    rows = read_rows(args.outdir/"validation/endpoint_refinements.csv")
    pairs = {(r["variant"], r["coupling"], float(r["k"]), r["readout"]): r
             for r in rows if r["study"] == "pair"}
    populations = {(r["variant"], r["grid"], int(r["population_N"]), r["readout"]): r
                   for r in rows if r["study"] == "population" and r["condition"] == "coupled"}
    pair_k = max((k for _, _, k, _ in pairs), default=10.)
    population_n = max((n for variant, grid, n, _ in populations
                        if variant != "reference" and grid == "reference"), default=50)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE), plt.rc_context({"font.size": 11, "axes.labelsize": 13,
            "xtick.labelsize": 10, "ytick.labelsize": 11, "axes.linewidth": .8,
            "svg.hashsalt": "figure8-validation"}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), layout="constrained")
        for ax, readout, letter in zip(axes[0], READOUTS, "ab"):
            for i, (variant, label) in enumerate(REFINEMENTS):
                for coupling, offset in zip(COUPLINGS, np.linspace(-.24, .24, 4)):
                    row = pairs.get((variant, coupling, pair_k, readout))
                    point(ax, i+offset, row, MARKERS[coupling], COLORS[coupling], 1e3)
            ax.set_xticks(range(4), [label for _, label in REFINEMENTS])
            ax.set_ylabel(r"$10^3\Delta N_{\mathrm{corr}}$")
            ax.set_title(f"({letter})  Pairs: {SIGNALS[readout][2]}, "+rf"$k={pair_k:g}\tau_r^{{-1}}$",
                         loc="left", fontsize=12)

        c, d = axes[1]
        for i, (variant, label) in enumerate(REFINEMENTS[1:]):
            for readout, offset in zip(READOUTS, (-.12, .12)):
                row = populations.get((variant, "reference", population_n, readout))
                point(c, i+offset, row, *SIGNALS[readout][:2])
        c.set_xticks(range(3), [label for _, label in REFINEMENTS[1:]])
        c.set_ylabel(r"$\Delta N_{\mathrm{corr}}$")
        c.set_title(rf"(c)  Coupled population: $N={population_n}$", loc="left", fontsize=12)

        sizes = sorted({n for variant, grid, n, _ in populations
                        if variant == "reference" and grid == "fine_lags"})
        for i, n in enumerate(sizes):
            for readout, offset in zip(READOUTS, (-.12, .12)):
                row = populations.get(("reference", "fine_lags", n, readout))
                point(d, i+offset, row, *SIGNALS[readout][:2], scale=1e3)
        d.set_xticks(range(len(sizes)), [str(n) for n in sizes])
        d.set(xlabel=r"$N$", ylabel=r"$10^3\Delta N_{\mathrm{corr}}$")
        d.set_title("(d)  Coupled population: finer lag grid", loc="left", fontsize=12)
        for ax in axes.flat:
            ax.axhline(0, color=".6", lw=.8, ls="--", zorder=0)
            ax.margins(x=.12)
        handles = [Line2D([], [], marker=MARKERS[k], color=COLORS[k], mfc="white",
                          ls="", label=LABELS[k]) for k in COUPLINGS]
        fig.legend(handles=handles, loc="outside upper center", ncols=4,
                   frameon=False, fontsize=11)
        handles = [Line2D([], [], marker=m, color=color, mfc="white", ls="", label=label)
                   for m, color, label in SIGNALS.values()]
        fig.legend(handles=handles, frameon=False, fontsize=11,
                   loc="outside lower center", ncols=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
