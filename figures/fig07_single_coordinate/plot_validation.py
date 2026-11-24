"""Plot Figure 7 EPR and effective frequency changes under simulation refinements."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D

from epr.plot_figure7 import PUBLICATION_STYLE

HERE = Path(__file__).resolve().parent
EPR_CASES = ('r004', 'r059', 'r135', 'r098', 'r088', 'r095')
FREQUENCY_CASES = ('r009', 'r015', 'r107', 'r111', 'r060', 'r100')
CHANGES = (('timestep', 'dt', .5, '#9dbdd2', 'o', r'$\Delta t/2$'),
           ('timestep', 'dt', .25, '#2c678f', 's', r'$\Delta t/4$'),
           ('burn_in', 'burn', 2, 'lightcoral', 'D', r'Equilibration $\times2$'),
           ('measurement_duration', 'measure', None, 'sandybrown', '^',
            r'Duration $\times2$ (a), $\times4$ (b)'))


def read(root, name):
    return json.loads((root/'data'/name).read_text())


def configs(plan):
    return {row['id']: {**plan['run_defaults'], **row} for row in plan['runs']}


def parameter(value):
    return r'10/3' if value == 10/3 else f'{value:g}'


def refinements(ax, rows, runs, cases, quantity, duration_factor, fields):
    ax.axvline(0, color='.6', lw=.8, ls='--')
    for offset, (changed, field, factor, color, marker, label) in zip(
            np.linspace(-.27, .27, len(CHANGES)), CHANGES):
        factor = duration_factor if factor is None else factor
        for row in rows:
            identifier = row['baseline_run']
            if identifier not in cases or row['changed'] != changed:
                continue
            base, refined = runs[identifier], runs[row['refined_run']]
            if not np.isclose(refined[field]/base[field], factor):
                continue
            ax.errorbar(1e3*row[quantity+'_change'], cases.index(identifier)+offset,
                        xerr=1e3*row[quantity+'_change_ci95_half_width'], fmt=marker,
                        color=color, mfc='white', ms=4, lw=1, capsize=2)
    labels = ['$('+','.join(parameter(runs[key][f]) for f in fields)+')$' for key in cases]
    ax.set_yticks(range(len(cases)), labels=labels)
    ax.set_ylim(len(cases)-.5, -.5)
    ax.tick_params(axis='y', length=0, pad=7)
    ax.margins(x=.08)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, help='EPR run directory containing data/')
    parser.add_argument('--dynamics-outdir', type=Path, default=HERE/'frequency',
                        help='Effective frequency run directory containing data/')
    parser.add_argument('--output', type=Path, help='Destination SVG (default: OUTDIR/validation.svg)')
    args = parser.parse_args()
    epr_dir = args.outdir or HERE/'epr'
    output = args.output or (args.outdir or HERE)/'validation.svg'
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epr = read(epr_dir, 'validation.json')
    frequency = read(args.dynamics_outdir, 'validation.json')
    with plt.rc_context(PUBLICATION_STYLE), plt.rc_context({'font.size': 11,
            'axes.labelsize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
            'axes.linewidth': .8, 'xtick.major.size': 4, 'ytick.major.size': 4,
            'svg.hashsalt': 'figure7-validation'}):
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
        refinements(a, epr['simulation_refinements'], configs(read(epr_dir, 'parameters.json')),
                    EPR_CASES, 'stratonovich_epr', 2, ('number', 'beta', 'k'))
        a.set(xlabel=r'$10^3(\sigma_{\mathrm{refined}}-\sigma_{\mathrm{base}})/(N\sigma_0)$',
              ylabel=r'$(N,\beta,k)$', title='(a)  EPR refinements')
        refinements(b, frequency['simulation_refinements'], configs(read(args.dynamics_outdir, 'parameters.json')),
                    FREQUENCY_CASES, 'frequency', 4, ('number', 'beta', 'omega', 'k'))
        b.set(xlabel=r'$10^3(\omega_{\mathrm{eff,refined}}-\omega_{\mathrm{eff,base}})/\omega$',
              ylabel=r'$(N,\beta,\omega,k)$', title='(b)  Effective frequency refinements')
        handles = [Line2D([], [], marker=marker, mfc='white', ls='', color=color, label=label)
                   for _, _, _, color, marker, label in CHANGES]
        fig.legend(handles=handles, loc='outside upper center', ncols=4, frameon=False,
                   fontsize=10)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, metadata={'Date': None})
        plt.close(fig)


if __name__ == '__main__':
    main()
