"""Plot changes in Figure 12 quantities under radial and coupling-grid refinements."""

import argparse
import csv
from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D

from figure12 import MAPS, STYLE

HERE = Path(__file__).resolve().parent
COLORS = ("#b2182b", "#777777", "#2166ac")
COMPARISONS = (("radial_resolution", "-", "Radial grid"),
               ("coupling_step", "--", r"$k$ grid"))
LABELS = (r"$\langle r\rangle_0^{-1}\partial_k\langle r\rangle_k$",
          r"$\langle r\rangle_0^{-1}\partial_k\langle r\rangle_k$",
          r"$\operatorname{Var}_0(r)^{-1}\partial_k\operatorname{Var}_k(r)$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE, help="Run directory containing data/")
    parser.add_argument("--data", type=Path, help="Read a specific numerical comparison table")
    parser.add_argument("--output", type=Path, help="Destination SVG (default: OUTDIR/validation.svg)")
    args = parser.parse_args()
    data = args.data or args.outdir/"data/numerics.csv"
    output = args.output or args.outdir/"validation.svg"
    with data.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["comparison"] in {name for name, _, _ in COMPARISONS}]
    betas = sorted({float(r["beta"]) for r in rows})
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE), plt.rc_context({"font.size": 11,
            "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
            "svg.hashsalt": "figure12-validation"}):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
        for ax, quantity, label, letter in zip(axes, MAPS, LABELS, "abc"):
            for beta, color in zip(betas, COLORS):
                for comparison, style, _ in COMPARISONS:
                    group = sorted((r for r in rows if r["quantity"] == quantity
                                    and r["comparison"] == comparison
                                    and float(r["beta"]) == beta), key=lambda r: float(r["k"]))
                    if group:
                        ax.plot([float(r["k"]) for r in group],
                                [float(r["absolute_difference"]) for r in group],
                                color=color, ls=style, lw=1.2)
            ax.set(xlabel=r"$k\;(\tau_r^{-1})$", yscale="log")
            ax.set_title(f"({letter})  {label}", fontsize=12)
        axes[0].set_ylabel("Absolute change")
        handles = [Line2D([], [], color=color, label=rf"$\beta={beta:.5g}$")
                   for beta, color in zip(betas, COLORS)]
        handles += [Line2D([], [], color=".25", ls=style, label=label)
                    for _, style, label in COMPARISONS]
        fig.legend(handles=handles, loc="outside upper center", ncols=5,
                   frameon=False, fontsize=11)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
