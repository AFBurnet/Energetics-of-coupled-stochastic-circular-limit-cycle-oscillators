"""Plot the saved Figure 3 EPR comparisons and simulation refinements."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

from figure3 import load_parameters, publication_style, run_configs

HERE = Path(__file__).resolve().parent
ESTIMATORS = {'stratonovich_epr': ('Stratonovich', 'x'),
              'radial_moment_epr': ('Radial moments', 'o')}
COLORS = {2: 'lightcoral', 4: 'sandybrown', 8: '#a6cee3', 16: '#377eb8'}


def interval(ax, x, y, half_width, marker, color, *, horizontal=False):
    errors = {'xerr' if horizontal else 'yerr': 1e3*half_width}
    ax.errorbar(x, y, **errors, fmt=marker, color=color, mfc='white',
                ms=4.5, lw=1, capsize=2)


def plot_benchmarks(ax, rows):
    ax.axvline(0, color='.6', lw=.8, ls='--')
    for i, row in enumerate(rows):
        for offset, (name, (_, marker)) in zip((-.13, .13), ESTIMATORS.items()):
            difference = row[name]-row['quadrature_epr']
            interval(ax, 1e3*difference, i+offset, row[name+'_ci95_half_width'],
                     marker, COLORS[row['number']], horizontal=True)
    labels = [rf"$({row['number']},\,{row['k']:g})$" for row in rows]
    ax.set_yticks(range(len(rows)), labels=labels)
    ax.invert_yaxis()
    ax.set(xlabel=r'$10^3(\sigma-\sigma_{\mathrm{quad}})/(N\sigma_0)$', ylabel=r'$(N,k)$')


def plot_refinements(ax, rows, configs, cases, changes):
    ax.axhline(0, color='.6', lw=.8, ls='--')
    offsets = dict(zip(cases, np.linspace(-.29, .29, len(cases))))
    for x, (changed, field, factor, label) in enumerate(changes):
        for row in rows:
            base, refined = configs[row['baseline_run']], configs[row['refined_run']]
            if row['changed'] != changed or not np.isclose(refined[field]/base[field], factor):
                continue
            position = x+offsets[row['number'], row['k']]
            for offset, (name, (_, marker)) in zip((-.022, .022), ESTIMATORS.items()):
                interval(ax, position+offset, 1e3*row[name+'_change'],
                         row[name+'_change_ci95_half_width'], marker, COLORS[row['number']])
    ax.set_xticks(range(len(changes)), labels=[item[3] for item in changes])
    ax.set_ylabel(r'$10^3(\sigma_{\mathrm{refined}}-\sigma_{\mathrm{base}})/(N\sigma_0)$')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=HERE, help='Run directory containing data/')
    parser.add_argument('--output', type=Path, help='Destination SVG (default: OUTDIR/validation.svg)')
    args = parser.parse_args()
    output = args.output or args.outdir/'validation.svg'
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plan = load_parameters(args.outdir)
    configs = run_configs(plan)
    checks = json.loads((args.outdir/'data/validation.json').read_text())
    refinements = checks['timestep_burn_duration_refinements']
    baseline_ids = {row['baseline_run'] for row in refinements}
    benchmarks = sorted((row for row in checks['estimator_benchmarks'] if row['run_id'] in baseline_ids),
                        key=lambda row: (row['number'], row['k']))
    cases = [(row['number'], row['k']) for row in benchmarks]
    timestep = [('timestep', 'dt', factor, label) for factor, label in
                ((2, r'$2\Delta t$'), (.5, r'$\Delta t/2$'), (.25, r'$\Delta t/4$'))]
    duration = [('burn_in', 'burn', 2, r'Equilibration $\times 2$'),
                ('measurement_duration', 'measure', 2, r'Duration $\times 2$')]

    with publication_style(plt), plt.rc_context({
            'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
            'svg.hashsalt': 'figure3-validation'}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
        plot_benchmarks(axes[0], benchmarks)
        plot_refinements(axes[1], refinements, configs, cases, timestep)
        plot_refinements(axes[2], refinements, configs, cases, duration)
        for letter, ax, title in zip('abc', axes,
                ('EPR benchmarks', 'Timestep refinements', 'Equilibration and duration')):
            ax.set_title(f'({letter})  {title}', loc='left', fontsize=12)
        handles = [Line2D([], [], marker=marker, mfc='white', ls='', color='.2', label=label)
                   for label, marker in ESTIMATORS.values()]
        fig.legend(handles=handles, loc='outside upper center', ncols=2, frameon=False,
                   title=rf"$\beta={plan['run_defaults']['beta']:g}$")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, metadata={'Date': None})
        plt.close(fig)


if __name__ == '__main__':
    main()
