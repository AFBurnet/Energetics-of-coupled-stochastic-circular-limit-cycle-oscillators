#!/usr/bin/env python3
"""Compute and plot the stationary Cartesian radial deformation maps."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

import numerics

HERE = Path(__file__).resolve().parent
PANEL_COLORS = (
    (-0.7092134810627083, 0.0487764045507042, ('-0.7092', '0.0000', '0.0488')),
    (-0.0011264512043248928, 9.224529223342888e-05, ('-0.001126', '0.000', '0.000092')),
    (-0.7123524739770364, 0.46951851482399953, ('-0.712', '0.000', '0.470')),
)
GRIDS = ('beta_grid', 'small_k_grid', 'large_k_grid')
MAPS = ('small_mean_gradient', 'large_mean_gradient', 'small_variance_gradient')
STYLE = {'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsmath}',
         'font.family': 'serif', 'font.serif': ['Computer Modern Roman'], 'font.size': 10,
         'svg.hashsalt': 'figure12',
         'axes.labelsize': 12, 'axes.linewidth': .8, 'axes.grid': False,
         'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
         'xtick.major.width': .8, 'ytick.major.width': .8}


def load_settings(root):
    return json.loads((Path(root) / 'data' / 'settings.json').read_text())


def grid(spec):
    return np.linspace(spec['start'], spec['stop'], spec['points'])


def load_data(outdir):
    data = Path(outdir)/'data'
    settings = load_settings(outdir)
    with np.load(data/'reference.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    return arrays, settings


def plot(outdir=HERE, dpi=180):
    arrays, _ = load_data(outdir)
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'cartesian_deformation_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(362 / 72, 287 / 72), dpi=dpi)
        rectangles = ((73, 37, 252, 258), (513, 37, 254, 258),
                      (293.5, 382, 252, 258))
        formulas = (
            r'$\displaystyle\frac{1}{\langle r\rangle_0}\frac{\mathrm{d}}{\mathrm{d}k}\langle r\rangle_k$',
            r'$\displaystyle\frac{1}{\langle r\rangle_0}\frac{\mathrm{d}}{\mathrm{d}k}\langle r\rangle_k$',
            r'$\displaystyle\frac{1}{\mathrm{Var}_0(r)}\frac{\mathrm{d}}{\mathrm{d}k}\mathrm{Var}_k(r)$')
        panels = (
            (arrays['small_ks'], arrays['small_mean_gradient'], [0, 1, 2, 3]),
            (arrays['large_ks'], arrays['large_mean_gradient'], [20, 25, 30, 35, 40]),
            (arrays['small_ks'], arrays['small_variance_gradient'], [0, 1, 2, 3]),
        )
        betas = arrays['betas']
        for letter, rect, formula, panel, colors in zip('abc', rectangles, formulas, panels, PANEL_COLORS):
            x, y, width, height = rect
            bounds = [x / 905, 1 - (y + height) / 718, width / 905, height / 718]
            ax = fig.add_axes(bounds)
            ks, field, yticks = panel
            low, high, labels = colors
            norm = TwoSlopeNorm(vmin=low, vcenter=0, vmax=high)
            im = ax.imshow(field.T, origin='lower', aspect='auto', cmap='RdBu_r',
                           norm=norm, extent=[betas[0], betas[-1], ks[0], ks[-1]],
                           interpolation='hanning', interpolation_stage='rgba')
            ax.set(xlabel=r'$\beta$', xticks=[10, 20], yticks=yticks)
            ax.set_ylabel(r'$k\;(\tau_r^{-1})$', rotation=90, labelpad=0, fontsize=10)
            ax.text(.02, .96, f'({letter})', transform=ax.transAxes, va='top', fontsize=12)
            ax.text(.52, .71, formula, transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            colorbar_gap = 10 if letter == 'b' else 16
            cax = fig.add_axes([(x + width + colorbar_gap) / 905, bounds[1], 12 / 905, bounds[3]])
            bar = fig.colorbar(im, cax=cax, ticks=[low, 0, high])
            bar.ax.set_yticklabels([f'${value}$' for value in labels], fontsize=8)
            bar.ax.tick_params(pad=2)
        folder = Path(outdir) / 'figures'
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in ('png', 'svg', 'eps'):
            target = folder / f'figure12.{suffix}'
            fig.savefig(target, dpi=dpi, facecolor='white')
            print(target.resolve())
        plt.close(fig)


def compute(outdir, beta_points=None, k_points=None):
    """Calculate a fresh grid in a new directory."""
    settings = load_settings(HERE)
    for name, points in zip(GRIDS, (beta_points, k_points, k_points)):
        if points is not None:
            if points < 2:
                raise ValueError('At least two beta and coupling points are required.')
            settings[name]['points'] = points
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    data = outdir / 'data'
    data.mkdir()
    betas, small_ks, large_ks = (grid(settings[name]) for name in GRIDS)
    arrays = numerics.compute(betas, small_ks, large_ks,
                              settings['radial_points'])
    target = data / 'reference.npz'
    np.savez_compressed(target, **arrays)
    (data / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(target)


def validate(root):
    """Replay saved rows and compare radial and coupling-grid refinements."""
    saved, settings = load_data(root)
    indices = np.unique([0, len(saved['betas'])//2, len(saved['betas'])-1])
    betas = saved['betas'][indices]
    small, large = saved['small_ks'], saved['large_ks']
    points = settings['radial_points']
    baseline = numerics.compute(betas, small, large, points)
    for quantity in MAPS:
        delta = np.max(np.abs(baseline[quantity]-saved[quantity][indices]))
        np.testing.assert_allclose(baseline[quantity], saved[quantity][indices],
                                   rtol=2e-10, atol=2e-13)
        print(f'{quantity}: maximum replay difference = {delta:.3g}', flush=True)

    rows = []
    def record(comparison, quantity, beta, k, base_setting, fine_setting, base, fine):
        rows.append(dict(comparison=comparison, quantity=quantity, beta=beta, k=k,
                         baseline_setting=base_setting, refined_setting=fine_setting,
                         baseline=base, refined=fine, absolute_difference=abs(fine-base)))

    radial = numerics.compute(betas, small, large, 2*points)
    print('Radial-grid refinement completed.', flush=True)
    fine_grids = [np.sort(np.r_[ks, (ks[:-1]+ks[1:])/2]) for ks in (small, large)]
    coupling = numerics.compute(betas, *fine_grids, points)
    for quantity in MAPS:
        ks = large if quantity == 'large_mean_gradient' else small
        for i, beta in enumerate(betas):
            for j, k in enumerate(ks):
                base = baseline[quantity][i, j]
                record('radial_resolution', quantity, beta, k, points, 2*points,
                       base, radial[quantity][i, j])
                record('coupling_step', quantity, beta, k, ks[1]-ks[0], (ks[1]-ks[0])/2,
                       base, coupling[quantity][i, 2*j])

    for beta in betas:
        reference = numerics.uncoupled_moments(beta, numerics.radius_limit(beta))
        for count in (points, 2*points):
            moments = numerics.radial_moments(beta, 0., count)
            for name, base, fine in zip(('mean_radius', 'radial_variance'), moments, reference):
                record('uncoupled_normalization', name, beta, 0., count, 'adaptive', base, fine)
    beta = float(betas[0])
    upper = numerics.radius_limit(beta)
    finite = numerics.uncoupled_moments(beta, upper)
    infinite = numerics.uncoupled_moments(beta, np.inf)
    for name, base, fine in zip(('mean_radius', 'radial_variance'), finite, infinite):
        record('radial_domain', name, beta, 0., upper, 'infinity', base, fine)
    path = Path(root)/'data/numerics.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} numerical comparisons in {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plot', 'validate', 'compute'))
    parser.add_argument('--outdir', type=Path, help='Results root (default: this folder for plot/check).')
    parser.add_argument('--dpi', type=float, default=180)
    parser.add_argument('--beta-points', type=int, help='Number of beta values (default: 100).')
    parser.add_argument('--k-points', type=int, help='Number of values per coupling grid (default: 100).')
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error('--dpi must be positive.')
    if args.command == 'compute':
        if args.outdir is None:
            parser.error('compute requires --outdir for a separate results directory.')
        compute(args.outdir, args.beta_points, args.k_points)
    elif args.command == 'plot':
        plot(args.outdir or HERE, args.dpi)
    else:
        validate(args.outdir or HERE)


if __name__ == '__main__':
    main()
