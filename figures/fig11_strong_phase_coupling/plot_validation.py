"""Plot Figure 11 trajectory refinements and comparisons with the stationary PDE."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from plotting import _pyplot, _style

HERE = Path(__file__).resolve().parent
CHANGES = (('dt_half', '#9dbdd2', 'o', r'$\Delta t/2$'),
           ('dt_quarter', '#2c678f', 's', r'$\Delta t/4$'),
           ('burn', 'lightcoral', 'D', r'Equilibration $\times2$'),
           ('duration', 'sandybrown', '^', r'Duration $\times2$'))


def errorbar(ax, interval, scale, position, **style):
    mean = scale*interval['mean']
    error = scale*np.array([[interval['mean']-interval['ci95_low']],
                            [interval['ci95_high']-interval['mean']]])
    ax.errorbar(mean, position, xerr=error, mfc='white', ms=4, lw=1,
                capsize=2, **style)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=HERE, help='Run directory with validation/report.json')
    parser.add_argument('--report', type=Path, help='Read this validation report instead')
    parser.add_argument('--output', type=Path, help='Output SVG (default: validation.svg in the run directory)')
    args = parser.parse_args()
    report = json.loads((args.report or args.outdir/'validation/report.json').read_text())
    output = args.output or args.outdir/'validation.svg'
    comparisons = report['asymptotic_comparison']
    k = [row['k'] for row in comparisons]
    pde = {row['k']: row['stationary_pde_epr'] for row in comparisons}
    plt = _pyplot()

    with plt.rc_context(_style()), plt.rc_context({'font.size': 11,
            'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
            'axes.linewidth': .8, 'xtick.major.size': 4, 'ytick.major.size': 4,
            'svg.hashsalt': 'figure11-validation'}):
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6), layout='constrained')
        a.axvline(0, color='.6', lw=.8, ls='--')
        for offset, (case, color, marker, _) in zip(np.linspace(-.27, .27, 4), CHANGES):
            for row in report['trajectory_refinements']:
                if row['case'] == case:
                    errorbar(a, row['stratonovich_epr_change'], 100/pde[row['k']],
                             k.index(row['k'])+offset, fmt=marker, color=color)
        a.set_yticks(range(len(k)), labels=[rf'${value:g}$' for value in k])
        a.set_ylim(len(k)-.5, -.5)
        a.set(xlabel=r'$(\sigma_{\mathrm{refined}}-\sigma_{\mathrm{base}})/\sigma_{\mathrm{PDE}}\;(\%)$',
              ylabel=r'$k\;(\tau_r^{-1})$', title='(a)  Trajectory refinements')
        a.legend(handles=[Line2D([], [], marker=marker, color=color, mfc='white', ls='',
                               label=label) for _, color, marker, label in CHANGES],
                 loc='upper center', bbox_to_anchor=(.5, -.19), ncols=2, frameon=False,
                 columnspacing=1)

        b.axhline(0, color='.6', lw=.8, ls='--')
        for index, row in enumerate(report['estimator_benchmarks']):
            interval, reference = row['stratonovich_epr'], pde[row['k']]
            mean = interval['mean']
            b.errorbar(row['k'], 100*(mean/reference-1),
                       yerr=100*np.array([[mean-interval['ci95_low']],
                                         [interval['ci95_high']-mean]])/reference,
                       fmt='o', color='#287C8E', mfc='white', ms=4, lw=1, capsize=2,
                       label='Stratonovich' if index == 0 else None)
        for field, color, linestyle, label in (
                ('leading', '#B08B4F', '--', r'$\mathcal{O}(k\ln k)$'),
                ('two_term', '#586B89', '-', r'$\mathcal{O}(k\ln k)+\mathcal{O}(k)$')):
            b.plot(k, [100*row[field+'_relative_difference'] for row in comparisons],
                   marker='.', color=color, ls=linestyle, lw=1.3, label=label)
        b.set(xscale='log', xlabel=r'$k\;(\tau_r^{-1})$',
              ylabel=r'$(\sigma-\sigma_{\mathrm{PDE}})/\sigma_{\mathrm{PDE}}\;(\%)$',
              title='(b)  Comparison with stationary PDE')
        b.set_xticks(k, labels=[rf'${value:g}$' for value in k], minor=False)
        b.set_xticks([], minor=True)
        b.legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncols=2, frameon=False,
                 columnspacing=1)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, metadata={'Date': None})
        plt.close(fig)


if __name__ == '__main__':
    main()
