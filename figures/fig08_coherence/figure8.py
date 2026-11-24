"""Plot Figure 8 from a single point table, calculate new cases, or check saved errors."""
import argparse
import csv
import os
from pathlib import Path
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
COUPLINGS = ("radial", "phase", "cartesian", "cartesian_x")
COLORS = dict(radial="#207c80", phase="#b85e40", cartesian="#6773a1", cartesian_x="#9b7e38")
MARKERS = dict(zip(COUPLINGS, ("o", "s", "D", "^")))
LABELS = dict(zip(COUPLINGS, ("Radial", "Phase", "full Cartesian", "single Cartesian")))
POP_COLORS = {"coupled": COLORS["cartesian"], "uncoupled": "#646464"}
READOUTS = ("oscillator_1", "collective")
STYLE = {"font.size": 8, "axes.titlesize": 11, "axes.labelsize": 9,
         "legend.fontsize": 7, "svg.fonttype": "none", "text.usetex": True,
         "font.family": "serif", "font.serif": ["Computer Modern Roman"],
         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": True,
         "axes.spines.right": True, "axes.linewidth": .65, "xtick.direction": "in",
         "ytick.direction": "in", "xtick.major.size": 3, "ytick.major.size": 3,
         "xtick.major.width": .65, "ytick.major.width": .65, "axes.grid": False,
         "savefig.dpi": 300}


def read_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def bounds(row):
    if row is None or str(row.get("plot_included", "")).lower() == "false":
        return (np.nan, np.nan, np.nan)
    value = float(row["value"])
    se = float(row["standard_error"])
    low, high = value-se, value+se
    if not (np.isfinite([value, low, high]).all() and low <= value <= high):
        raise ValueError("A plotted point has nonfinite or unordered error bars.")
    return value, low, high


def point(axis, x, values, color, marker, *, hollow=False, size=3):
    value, low, high = values
    if np.isfinite(value):
        axis.errorbar(x, value, yerr=[[value-low], [high-value]], linestyle="none",
                      marker=marker, color=color, markerfacecolor="white" if hollow else color,
                      markeredgewidth=.65, markersize=size, capsize=1.3, elinewidth=.6, zorder=4)


def frame(axis, panel, *, pair):
    axis.set_box_aspect(1)
    axis.tick_params(top=True, right=True, pad=2.)
    axis.grid(False, which="both", axis="both")
    x, y, align = {"a": (.86, .96, "right"), "e": (.04, .84, "left"),
                   "f": (.91, .96, "right")}.get(panel, (.04, .96, "left"))
    axis.text(x, y, rf"$\mathrm{{({panel})}}$", transform=axis.transAxes,
              fontsize=9, ha=align, va="top")
    axis.set_xlabel(r"$k\;(\tau_r^{-1})$" if pair else r"$N$", labelpad=1.5)


def plot_pair(axes, rows):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    strengths = sorted({float(r["k"]) for r in rows})
    lookup = {(r["panel"], r["readout"], r["coupling"], float(r["k"])): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate rows in the pair point table.")
    panels = (("a", "total_system"), ("b", READOUTS[0]), ("c", READOUTS[1]),
              ("d", READOUTS[0]), ("d", READOUTS[1]))
    cycle_limits = []
    for coupling in COUPLINGS:
        for panel, readout in panels:
            axis = axes[panel]
            base = bounds(lookup.get((panel, readout, "shared_uncoupled", 0.)))
            if not any((panel, readout, coupling, k) in lookup for k in strengths):
                continue
            color, marker = COLORS[coupling], MARKERS[coupling]
            hollow = panel == "d" and readout == READOUTS[1]
            values = [base if k == 0 else bounds(lookup.get((panel, readout, coupling, k)))
                      for k in strengths]
            for k, val in zip(strengths, values):
                if k:
                    point(axis, k, val, color, marker, hollow=hollow)
                    if panel in "bc" and np.isfinite(val[0]):
                        cycle_limits.extend(val[1:])
            axis.plot(strengths, [v[0] for v in values], color=color,
                      linestyle="--" if hollow else "-", linewidth=.85, zorder=2)
    for panel, readout in panels[:-1]:
        axis = axes[panel]
        values = bounds(lookup.get((panel, readout, "shared_uncoupled", 0.)))
        point(axis, 0., values, "#333333", "*", size=5.5)
        if panel in "ad":
            axis.axhline(1., color="#999999", linestyle=":", linewidth=.55)
        elif np.isfinite(values[0]):
            cycle_limits.extend(values[1:])
    for panel, label in zip("abcd", (r"$\sigma\;(2\sigma_0)$", r"$N_{\mathrm{corr}}^{(1)}$",
                                 r"$N_{\mathrm{corr}}^{(Z)}$", r"$A_{\mathrm{rms}}(k)/A_{\mathrm{rms}}(0)$")):
        axis = axes[panel]
        frame(axis, panel, pair=True)
        axis.set_ylabel(label, labelpad=2.)
        if len(strengths) <= 5:
            axis.set_xticks(strengths)
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        span = max(strengths) - min(strengths)
        axis.set_xlim(strengths[0]-.06*span, strengths[-1]+.06*span)
    if cycle_limits:
        pad = .12 * np.ptp(cycle_limits) or .01
        for panel in "bc":
            axes[panel].set_ylim(min(cycle_limits)-pad, max(cycle_limits)+pad)
    readout_handles = [Line2D([], [], color="#444444", marker="o", markersize=2.8,
                             linestyle=line, linewidth=.8, markerfacecolor=face, label=label)
                       for face, line, label in (("#444444", "-", "Individual"),
                                                ("white", "--", "Collective"))]
    axes["d"].legend(handles=readout_handles, loc="center right", bbox_to_anchor=(.98, .385),
                     frameon=False, fontsize=5.5, borderaxespad=0, borderpad=0,
                     handlelength=1.2, labelspacing=.1, handletextpad=.35)
    plotted_couplings = {row["coupling"] for row in rows}
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], markersize=2.8,
                      linewidth=.85, label=LABELS[c]) for c in COUPLINGS if c in plotted_couplings]
    handles.append(Line2D([], [], color="#333333", marker="*", markersize=5,
                           linestyle="none", label="Uncoupled"))
    axes["a"].legend(handles=handles, loc="upper left", frameon=False, fontsize=5.5,
                     borderaxespad=.3, handlelength=1., labelspacing=.15, handletextpad=.4)


def plot_population(axes, rows):
    from matplotlib.lines import Line2D

    sizes = sorted({int(r["population_N"]) for r in rows})
    K, = {float(r["K"]) for r in rows}
    lookup = {(r["panel"], r["readout"], r["condition"], int(r["population_N"])): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate rows in the population point table.")
    for panel, readouts in (("e", ("total_system",)), ("f", READOUTS)):
        axis = axes[panel]
        for condition in ("coupled", "uncoupled"):
            for readout in readouts:
                values = np.array([bounds(lookup.get((panel, readout, condition, n))) for n in sizes])
                hollow, color = readout == "collective", POP_COLORS[condition]
                no_markers = panel == "e" and condition == "uncoupled"
                axis.plot(sizes, values[:, 0], linestyle="--" if hollow else "-", color=color,
                          linewidth=.85, zorder=2, label="Uncoupled" if no_markers else None)
                valid = np.isfinite(values).all(axis=1)
                if not no_markers and valid.any():
                    value, lo, hi = values[valid].T
                    axis.errorbar(np.array(sizes)[valid], value, yerr=[value-lo, hi-value],
                                  linestyle="none", color=color, marker="o", markersize=3.,
                                  markerfacecolor="white" if hollow else color, markeredgewidth=.65,
                                  capsize=1.3, elinewidth=.6, zorder=4,
                                  label="full Cartesian" if panel == "e" else None)
        frame(axis, panel, pair=False)
        axis.set_xticks(sizes)
        pad = .045*(sizes[-1]-sizes[0]) or 1.
        axis.set_xlim(sizes[0]-pad, sizes[-1]+pad)
        if len(sizes) > 1:
            axis.get_xticklabels()[0].set_ha("right")
        if panel == "e":
            axis.set_ylabel(r"$\sigma\;(N\sigma_0)$", labelpad=2.)
            handles, labels = axis.get_legend_handles_labels()
            order = [labels.index(label) for label in ("full Cartesian", "Uncoupled") if label in labels]
            axis.legend([handles[i] for i in order], [labels[i] for i in order],
                        loc="center right", bbox_to_anchor=(.98, .65), frameon=False,
                        fontsize=5.5, handlelength=1.2, handletextpad=.4)
            axis.margins(y=.12)
        else:
            axis.set_ylabel(r"$N_{\mathrm{corr}}$", labelpad=2.)
            axis.set_ylim(bottom=0.)
            axis.margins(y=.16)
            handles = [Line2D([], [], color=POP_COLORS[condition], marker="o", markersize=3.,
                               linewidth=.85, linestyle=line, markerfacecolor=face,
                               markeredgewidth=.65, label=condition_label+": "+readout)
                       for condition, condition_label in (("coupled", "full Cartesian"), ("uncoupled", "Uncoupled"))
                       for line, face, readout in (("-", POP_COLORS[condition], "indiv."), ("--", "white", "coll."))]
            axis.legend(handles=handles, loc="upper left", frameon=False, fontsize=5.,
                        borderaxespad=.35, handlelength=1.2, handletextpad=.35, labelspacing=.2)
        x, y = (.462, .54) if panel == "e" else (.04, .70)
        axis.text(x, y, rf"$K={K:g}$", transform=axis.transAxes, fontsize=6.5, ha="left", va="top")


def plot(rows, outdir):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"coherence_mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cairosvg

    pair = [r for r in rows if r["panel"] in "abcd"]
    population = [r for r in rows if r["panel"] in "ef"]
    if not (pair or population):
        raise ValueError("The point table has no Figure 8 panels.")
    panels = ("abcd" if pair else "") + ("ef" if population else "")
    outdir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        height = {2: 2.25, 4: 3.8, 6: 5.4}[len(panels)]
        fig, axes = plt.subplots(len(panels)//2, 2, figsize=(3.8, height))
        axes = dict(zip(panels, axes.flat))
        if pair:
            plot_pair(axes, pair)
        if population:
            plot_population(axes, population)
        fig.subplots_adjust(left=.17, right=.985, bottom=.075 if population else .11,
                            top=.955 if population else .935, wspace=.31, hspace=.28)
        for extension in ("png", "svg"):
            fig.savefig(outdir/f"figure.{extension}", bbox_inches="tight", pad_inches=.02)
        cairosvg.svg2eps(url=str(outdir/"figure.svg"), write_to=str(outdir/"figure.eps"))
        plt.close(fig)
    print(f"Wrote {outdir/'figure.png'} (also SVG and EPS).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plot_parser = sub.add_parser("plot", help="plot saved point estimates and one-SE bars")
    plot_parser.add_argument("--table", type=Path, nargs="+", default=[HERE/"data/figure.csv"])
    plot_parser.add_argument("--outdir", type=Path, default=HERE/"figures")
    validation = sub.add_parser("validate", help="check saved or recalculated bootstrap estimates")
    validation.add_argument("--outdir", type=Path, default=HERE, help="Calculation directory")
    from compute import add_compute_arguments
    add_compute_arguments(sub.add_parser("compute", help="run coherence, EPR and amplitude calculations"))
    args = parser.parse_args()
    if args.command == "plot":
        plot([r for path in args.table for r in read_rows(path)], args.outdir)
    elif args.command == "validate":
        from validate import validate, validate_calculation
        if args.outdir.resolve() == HERE:
            validate(HERE)
        else:
            validate_calculation(args.outdir)
    else:
        from compute import compute
        compute(args, HERE, COUPLINGS)


if __name__ == "__main__":
    main()
