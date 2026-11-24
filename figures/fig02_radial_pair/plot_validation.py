"""Plot the saved Figure 2 validation results."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

from figure2 import publication_style

HERE = Path(__file__).resolve().parent
ESTIMATORS = {
    'stratonovich_epr': ('Stratonovich', 'x'),
    'radial_moment_epr': ('Radial moments', 'o'),
}
COLORS = {10 / 3: '#b0c4de', 7.: '#c03029', 10.: '#f78a70'}


def interval(ax, x, y, half_width, marker, color, *, horizontal=False):
    errors = {'xerr' if horizontal else 'yerr': 1e3 * half_width}
    ax.errorbar(x, y, **errors, fmt=marker, color=color, mfc='white',
                ms=4.5, lw=1, capsize=2)


def plot_benchmarks(ax, rows):
    ax.axvline(0, color='.6', lw=.8, ls='--')
    for i, row in enumerate(rows):
        for offset, (key, (_, marker)) in zip((-.13, .13), ESTIMATORS.items()):
            difference = row[key] - row['quadrature_epr']
            interval(ax, 1e3 * difference, i + offset, row[key + '_ci95_half_width'],
                     marker, COLORS[row['beta']], horizontal=True)
    labels = []
    for row in rows:
        beta = '10/3' if row['beta'] == 10 / 3 else f"{row['beta']:g}"
        labels.append(rf"$({beta},\,{row['k']:.3g})$")
    ax.set_yticks(range(len(rows)), labels=labels)
    ax.invert_yaxis()
    ax.set(xlabel=r'$10^3(\sigma-\sigma_{\mathrm{quad}})/(2\sigma_0)$',
           ylabel=r'$(\beta,k)$')


def plot_refinements(ax, benchmarks, rows):
    changes = {'timestep': r'$\Delta t/2$', 'burn_in': r'Equilibration $\times 2$',
               'measurement_duration': r'Duration $\times 2$'}
    positions = {name: x for x, name in enumerate(changes)}
    cases = {(row['beta'], row['k']): x for row, x in
             zip(benchmarks, np.linspace(-.29, .29, len(benchmarks)))}
    ax.axhline(0, color='.6', lw=.8, ls='--')
    for row in rows:
        key, beta = row['estimator'], row['beta']
        x = positions[row['changed']] + cases[beta, row['k']]
        x += -.022 if key == 'stratonovich_epr' else .022
        interval(ax, x, 1e3 * row['refined_minus_baseline'],
                 row['difference_ci95_half_width'], ESTIMATORS[key][1], COLORS[beta])
    ax.set_xticks(range(len(changes)), labels=changes.values())
    ax.set_ylabel(r'$10^3(\sigma_{\mathrm{refined}}-\sigma_{\mathrm{base}})/(2\sigma_0)$')


def plot_depth(ax, minimum):
    ax.axhline(0, color='.6', lw=.8, ls='--')
    for x, (key, (_, marker)) in enumerate(ESTIMATORS.items()):
        row = minimum['depth'][key]
        interval(ax, x, 1e3 * row['depth'], row['ci95_half_width'], marker, COLORS[7.])
    ax.plot(2, 1e3 * minimum['quadrature_depth'], 's', color='.25', ms=4.5)
    ax.set_xticks(range(3), labels=[name for name, _ in ESTIMATORS.values()] + ['Quadrature'])
    ax.set(xlim=(-.45, 2.45), ylabel=r'$10^3[\sigma(0)-\sigma(k_*)]/(2\sigma_0)$')
    setting, = (row for row in minimum['long_run_estimates'] if row['case'] == 'minimum')
    ax.text(.98, .06, rf"$\beta={setting['beta']:g},\quad k_*={setting['k']:.3g}$",
            transform=ax.transAxes, ha='right', va='bottom')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=HERE, help='Directory containing the calculation results')
    parser.add_argument('--output', type=Path, help='Destination SVG, defaults to validation.svg in --outdir')
    args = parser.parse_args()
    output = args.output if args.output is not None else args.outdir/'validation.svg'
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    checks = json.loads((args.outdir/'data/validation.json').read_text())

    with publication_style(plt), plt.rc_context({
            'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
            'svg.hashsalt': 'figure2-validation'}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
        benchmarks = checks['estimator_benchmarks']
        plot_benchmarks(axes[0], benchmarks)
        plot_refinements(axes[1], benchmarks, checks['timestep_burn_duration_refinements'])
        plot_depth(axes[2], checks['shallow_minimum'])

        titles = ('EPR benchmarks', 'Trajectory refinements', 'Minimum depth')
        for letter, ax, title in zip('abc', axes, titles):
            ax.set_title(f'({letter})  {title}', loc='left', fontsize=12)
        handles = [Line2D([], [], marker=marker, mfc='white', ls='', color='.2', label=label)
                   for label, marker in ESTIMATORS.values()]
        fig.legend(handles=handles, loc='outside upper center', ncols=2, frameon=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, metadata={'Date': None})
        plt.close(fig)


if __name__ == '__main__':
    main()
