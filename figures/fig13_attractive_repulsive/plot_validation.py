"""Plot the Figure 13 phase-PDE sensitivity to radial resolution and domain."""

import argparse
import csv
from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D

from figure13 import STYLE

HERE = Path(__file__).resolve().parent
CHANGES = (('radial_resolution', '-', 'o', 'Radial grid'),
           ('radial_domain', '--', '^', 'Radial domain'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=HERE, help='Run directory with data/numerics.csv')
    parser.add_argument('--data', type=Path, help='Read this numerical comparison table instead')
    parser.add_argument('--output', type=Path, help='Output SVG (default: validation.svg in the run directory)')
    args = parser.parse_args()
    output = args.output or args.outdir/'validation.svg'
    with (args.data or args.outdir/'data/numerics.csv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['coupling'] == 'phase']
    detunings = sorted({float(row['delta_omega']) for row in rows})
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE), plt.rc_context({'font.size': 11,
            'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
            'svg.hashsalt': 'figure13-validation'}):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True, layout='constrained')
        for ax, delta, color, letter in zip(axes, detunings, ('#377EB8', '#DD5548'), 'ab'):
            for comparison, style, marker, _ in CHANGES:
                group = sorted((row for row in rows if float(row['delta_omega']) == delta
                                and row['comparison'] == comparison), key=lambda row: float(row['k']))
                ax.plot([float(row['k']) for row in group],
                        [float(row['absolute_difference']) for row in group],
                        color=color, ls=style, marker=marker, mfc='white', ms=4, lw=1.2)
            ax.set(xlabel=r'$k\;(\tau_r^{-1})$', yscale='log')
            ax.set_title(rf'({letter})  $\Delta\omega={delta:g}\,\tau_r^{{-1}}$', fontsize=12)
            ax.set_xticks(sorted({float(row['k']) for row in rows}))
        axes[0].set_ylabel(r'Absolute change in $\sigma/\sigma(0)$')
        handles = [Line2D([], [], color='.25', ls=style, marker=marker, mfc='white', label=label)
                   for _, style, marker, label in CHANGES]
        fig.legend(handles=handles, loc='outside upper center', ncols=2, frameon=False,
                   fontsize=11)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, metadata={'Date': None})
        plt.close(fig)


if __name__ == '__main__':
    main()
