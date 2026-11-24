"""Plot strong-coupling EPR and saved numerical sensitivity checks."""
from pathlib import Path
import os
import tempfile

import numpy as np


FIGURE_SIZE = (5.2, 3.8)
AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_SIZE = 18, 15, 13


def _pyplot():
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'phase_strong_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _style():
    return {
        'text.usetex': True,
        'legend.framealpha': 1,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.1,
        'ytick.major.width': 1.1,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
    }


def _save(fig, root, name):
    folder = Path(root) / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'eps'):
        fig.savefig(folder / f'{name}.{extension}', dpi=300,
                    transparent=False, facecolor='white')


def asymptotes(k, T_eff, theory):
    leading = theory['sigma_psi'] + theory['A'] * k * np.log(k / T_eff)
    return leading, leading + theory['B'] * k


def plot_main(root: Path, settings: dict, arrays: dict):
    """Plot strong-coupling EPR with one-SEM bars."""
    plt = _pyplot()
    from matplotlib.ticker import NullFormatter

    k = np.asarray(settings['k'], dtype=float)
    raw = np.stack([np.asarray(arrays[f'k{int(value)}_values']) for value in k])
    mean = raw.mean(axis=1)
    sem = raw.std(axis=1, ddof=1) / np.sqrt(raw.shape[1])
    pde = np.array([float(arrays[f'k{int(value)}_pde_epr']) for value in k])
    theory, sigma0 = settings['theory'], settings['sigma0']

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout='constrained')
        ax.errorbar(k, mean[:, 0] / sigma0, yerr=sem[:, 0] / sigma0,
                    fmt='o', color='#287C8E', markerfacecolor='white', markersize=5.5,
                    capsize=3, elinewidth=1, label='Stratonovich')
        ax.plot(k, pde / sigma0, 's-', color='#287C8E', markersize=3.5,
                linewidth=1.6, label='theory')
        tail = k[k >= settings['tail_from']]
        kk = np.geomspace(tail.min(), tail.max(), 250)
        leading, two_term = asymptotes(kk, settings['T_eff'], theory)
        ax.plot(kk, leading / sigma0, '--', color='#B08B4F', linewidth=1.7,
                label=r'$\mathcal{O}(k\ln k)$')
        ax.plot(kk, two_term / sigma0, '-', color='#586B89', linewidth=1.7,
                label=r'$\mathcal{O}(k\ln k)+\mathcal{O}(k)$')
        ax.set(xscale='log', yscale='log', xlabel=r'$k\ (\tau_r^{-1})$',
               ylabel=r'$\sigma\ (\sigma(0))$')
        ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.set_xticks(k, labels=[rf'${value:g}$' for value in k])
        ax.xaxis.set_minor_formatter(NullFormatter())
        lower, upper = ax.get_ylim()
        references = [(1., '#9CA9B5', ':')]
        if settings['delta_omega'] != 0:
            references.append((theory['sigma_psi']/sigma0, '#287C8E', '--'))
        for level, color, style in references:
            if lower <= level <= upper:
                ax.axhline(level, color=color, linestyle=style, linewidth=.8, zorder=0)
        ax.text(.98, .04, rf'$T_{{\mathrm{{eff}}}}={settings["T_eff"]:g},\ '
                rf'\bar\omega={settings["omega_bar"]:g},\ \Delta\omega={settings["delta_omega"]:g}$',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=14)
        ax.tick_params(colors='black', direction='out', which='both', labelsize=TICK_LABEL_SIZE)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(.8)
        ax.grid(False, which='both')
        ax.legend(loc='upper left', frameon=False, fontsize=LEGEND_SIZE)
        _save(fig, root, 'figure11')
        plt.close(fig)
