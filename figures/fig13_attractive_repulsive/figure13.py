#!/usr/bin/env python3
"""Stationary-theory comparison of attractive and repulsive pair coupling."""
import argparse
import csv
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import i0e

HERE = Path(__file__).resolve().parent


STYLE = {
    'text.usetex': True, 'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'], 'font.size': 9,
    'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'axes.spines.top': True, 'axes.spines.right': True,
    'axes.linewidth': .65, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.major.width': .75, 'ytick.major.width': .75,
    'axes.grid': False, 'axes.formatter.useoffset': False,
    'savefig.dpi': 600, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'svg.hashsalt': 'figure13',
}


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2) + '\n')


@lru_cache(None)
def free_moment(beta, power):
    """Continuum moment with the two-dimensional radial Jacobian."""
    def integrand(r, exponent):
        return r**exponent * np.exp(beta*(r*r/2-r**3/3-1/6))
    z = quad(integrand, 0, np.inf, args=(1,), epsabs=1e-12, epsrel=1e-12)[0]
    return quad(integrand, 0, np.inf, args=(power+1,),
                epsabs=1e-12, epsrel=1e-12)[0]/z


@lru_cache(None)
def quadrature_grid(order, rmax):
    x, w = leggauss(order)
    r, w = rmax*(x+1)/2, rmax*w/2
    rr, ss = r[:, None], r[None, :]
    base = rr*rr/2-rr**3/3 + ss*ss/2-ss**3/3
    log_measure = np.log(r*w)[:, None] + np.log(r*w)[None, :]
    return rr, ss, base, log_measure


def stationary_ratio(coupling, beta, k, order=128, rmax=6.):
    """Signed-k quadrature of the exact radial or full-Cartesian density.

    For equal intrinsic frequencies the EPR ratio is the coupled/uncoupled
    second-moment ratio. Cartesian angular integration gives I_0(beta*k*r*s),
    which is even in its argument; exp(abs(z))*i0e(z) remains stable for k<0.
    """
    rr, ss, base, measure = quadrature_grid(order, rmax)
    exponent = beta*base + measure
    if coupling == 'radial':
        exponent = exponent - beta*k*(rr-ss)**2/2
    elif coupling == 'cartesian':
        z = beta*k*rr*ss
        exponent = (exponent - beta*k*(rr*rr+ss*ss)/2
                    + np.abs(z) + np.log(i0e(z)))
    else:
        raise ValueError(f'Unknown coupling: {coupling}')
    weight = np.exp(exponent-exponent.max())
    m2 = np.sum(weight*(rr*rr+ss*ss)/2)/weight.sum()
    return float(m2/free_moment(beta, 2))


def weak_slope(coupling, beta):
    m1, m2, m3, m4 = [free_moment(beta, j) for j in range(1, 5)]
    covariance = m4-m2*m2
    if coupling == 'radial':
        covariance -= 2*m1*(m3-m1*m2)
    elif coupling != 'cartesian':
        raise ValueError(coupling)
    return float(-beta*covariance/(2*m2))


def read_settings(outdir):
    return json.loads((Path(outdir)/'data/settings.json').read_text())


def calculation_grids(settings):
    low, high = settings['k_range']
    return (np.linspace(low, high, settings['signed_points']),
            np.linspace(0., high, settings['phase_positive_points']))


def quadrature_case(coupling, beta, k, settings):
    return stationary_ratio(coupling, beta, k, order=settings['quadrature_order'],
                            rmax=settings['quadrature_rmax'])


def comparison(coupling, beta, delta, k, changed, base_setting, fine_setting, base, fine):
    return dict(coupling=coupling, beta=beta, delta_omega=delta, k=k,
                comparison=changed, baseline_setting=base_setting,
                refined_setting=fine_setting, baseline=base, refined=fine,
                absolute_difference=abs(fine-base))


def quadrature_refinements(settings):
    rows = []
    order, fine_order = settings['quadrature_order'], settings['refinement_order']
    radius, wide_radius = settings['quadrature_rmax'], settings['refinement_rmax']
    for coupling in ('radial', 'cartesian'):
        for beta in settings['betas']:
            for k in settings['quadrature_check_k']:
                base = stationary_ratio(coupling, beta, k, order, radius)
                fine = stationary_ratio(coupling, beta, k, fine_order, radius)
                wide = stationary_ratio(coupling, beta, k, fine_order, wide_radius)
                rows.extend((comparison(coupling, beta, 0., k, 'quadrature_order',
                                        order, fine_order, base, fine),
                             comparison(coupling, beta, 0., k, 'radial_domain',
                                        radius, wide_radius, fine, wide)))
    return rows


def phase_calculations(settings, positive):
    """Calculate the positive phase branch, weak coefficient and PDE refinements."""
    sys.path.insert(0, str(HERE.parent))
    from fig04_phase_coupling import numerics

    temperature, omega = 1./settings['phase_beta'], settings['phase_mean_frequency']
    meshes = phase_meshes(settings)

    @lru_cache(None)
    def epr(k, delta, mesh):
        return float(numerics.solve_pde(k, temperature, omega, delta, *mesh)['epr'])

    def ratio(k, delta, mesh):
        return epr(k, delta, mesh)/epr(0., delta, mesh)

    response = numerics.weak_pair_response(temperature, np.asarray(settings['phase_detunings']),
        nr=settings['phase_weak_nr'], rmax=settings['phase_weak_rmax'])
    curves, rows = [], []
    for j, delta in enumerate(settings['phase_detunings']):
        ratios = np.array([ratio(float(k), delta, meshes[0]) for k in positive])
        sigma_unc = (2*omega**2+delta**2/2)*response['m2']/temperature
        curves.append(dict(coupling='phase', beta=settings['phase_beta'], delta_omega=delta,
            k=np.r_[-positive[:0:-1], positive], ratio=np.r_[ratios[:0:-1], ratios],
            coefficient=float(response['C2'][j]/sigma_unc)))
        for k in settings['phase_check_k']:
            base, fine, wide, fourier = [ratio(k, delta, mesh) for mesh in meshes]
            domain = lambda mesh: f'nr={mesh[0]}, rmax={mesh[1]:g}'
            rows.extend((comparison('phase', settings['phase_beta'], delta, k,
                                    'radial_resolution', meshes[0][0], meshes[1][0], base, fine),
                         comparison('phase', settings['phase_beta'], delta, k,
                                    'radial_domain', domain(meshes[1]), domain(meshes[2]), fine, wide),
                         comparison('phase', settings['phase_beta'], delta, k,
                                    'fourier_modes', meshes[1][2], meshes[3][2], fine, fourier)))
        print(f'Phase detuning {delta:g}: curves and refinements complete.', flush=True)
    return curves, rows


def phase_meshes(settings):
    nr, rmax, modes = (settings[key] for key in ('phase_nr', 'phase_rmax', 'phase_modes'))
    fine_nr = settings['phase_refinement_nr']
    return ((nr, rmax, modes), (fine_nr, rmax, modes),
            (settings['phase_domain_nr'], settings['phase_domain_rmax'], modes),
            (fine_nr, rmax, settings['phase_refinement_modes']))


def write_comparisons(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute(outdir):
    """Recompute the stationary curves, weak coefficients and refinements."""
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    data = outdir/'data'
    data.mkdir()
    settings = read_settings(HERE)
    ks, positive = calculation_grids(settings)
    curves = []
    for coupling in ('radial', 'cartesian'):
        for beta in settings['betas']:
            curves.append(dict(coupling=coupling, beta=beta, delta_omega=0., k=ks,
                ratio=[quadrature_case(coupling, beta, k, settings) for k in ks],
                coefficient=weak_slope(coupling, beta)))
    phase, phase_checks = phase_calculations(settings, positive)
    curves.extend(phase)
    with (data/'curves.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['coupling', 'beta', 'delta_omega', 'k', 'normalized_epr'])
        for curve in curves:
            for k, ratio in zip(curve['k'], curve['ratio']):
                writer.writerow([curve['coupling'], curve['beta'], curve['delta_omega'], k, ratio])
    write_json(data/'weak_coefficients.json',
               [{key: c[key] for key in ('coupling', 'beta', 'delta_omega', 'coefficient')} for c in curves])
    write_json(data/'settings.json', settings)
    write_comparisons(data/'numerics.csv', quadrature_refinements(settings)+phase_checks)
    print(data/'curves.csv')


def load_curves(outdir):
    data = Path(outdir)/'data'
    coefficients = json.loads((data/'weak_coefficients.json').read_text())
    with (data/'curves.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    curves = []
    for entry in coefficients:
        selected = [row for row in rows if (row['coupling'], float(row['beta']), float(row['delta_omega']))
                    == (entry['coupling'], entry['beta'], entry['delta_omega'])]
        curves.append(dict(**entry, k=np.array([float(row['k']) for row in selected]),
                           ratio=np.array([float(row['normalized_epr']) for row in selected])))
    return curves


def validate(outdir=HERE):
    """Recalculate the retained stationary refinements and weak coefficients."""
    settings, curves = read_settings(outdir), load_curves(outdir)
    positive = np.unique([0., *settings['phase_check_k']])
    phase, phase_checks = phase_calculations(settings, positive)
    rows = quadrature_refinements(settings)+phase_checks
    largest = 0.
    for curve in curves:
        coupling, beta, delta = (curve[key] for key in ('coupling', 'beta', 'delta_omega'))
        if coupling == 'phase':
            fresh = next(c for c in phase if c['delta_omega'] == delta)
            ks, ratios, coefficient = fresh['k'], fresh['ratio'], fresh['coefficient']
        else:
            ks = settings['quadrature_check_k']
            ratios = [quadrature_case(coupling, beta, k, settings) for k in ks]
            coefficient = weak_slope(coupling, beta)
        indices = [np.argmin(abs(curve['k']-k)) for k in ks]
        np.testing.assert_allclose(curve['k'][indices], ks, rtol=0., atol=1e-12)
        np.testing.assert_allclose(curve['ratio'][indices], ratios, rtol=0., atol=1e-10)
        np.testing.assert_allclose(curve['coefficient'], coefficient, rtol=0., atol=1e-12)
        largest = max(largest, float(np.max(abs(curve['ratio'][indices]-ratios))))
    path = Path(outdir)/'data/numerics.csv'
    write_comparisons(path, rows)
    print(f'Curve replay: largest normalized EPR difference = {largest:.3g}.')
    print(f'Recalculated {len(rows)} numerical comparisons and {len(curves)} weak coefficients.')


def plot(outdir=HERE):
    curves = load_curves(outdir)
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'figure13_mpl'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'figure13_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    palettes = {'radial': ['#b0c4de', '#f78a70'], 'phase': ['#377EB8', '#DD5548'],
                'cartesian': ['#6b9cbd', '#15537e']}
    folder = Path(outdir)/'figures'
    folder.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.9), sharex=True)
        fig.subplots_adjust(left=.22, right=.95, top=.985, bottom=.135, hspace=.14)
        for ax, coupling, letter in zip(axes, ['radial', 'phase', 'cartesian'], 'abc'):
            ax.axhline(1., color='#999999', linestyle=':', linewidth=.55, zorder=1)
            ax.axvline(0., color='#b0b0b0', linestyle=':', linewidth=.5, zorder=1)
            selected = [c for c in curves if c['coupling'] == coupling]
            for color, curve in zip(palettes[coupling], selected):
                if coupling == 'phase':
                    label = rf'$\Delta\omega={curve["delta_omega"]:g}$'
                elif np.isclose(curve['beta'], 10/3):
                    label = r'$\beta=10/3$'
                else:
                    label = r'$\beta=10$'
                ax.plot(curve['k'], curve['ratio'], color=color, linewidth=1.25,
                        label=label, zorder=3)
                weak_k = np.linspace(-.07, .07, 81)
                weak = 1. + curve['coefficient']*weak_k**(2 if coupling == 'phase' else 1)
                ax.plot(weak_k, weak, color='#444444', linewidth=.7,
                        linestyle=(0, (3, 2.3)), zorder=4)
            name = {'radial': 'Radial', 'phase': 'Phase',
                    'cartesian': r'Full\ Cartesian'}[coupling]
            ax.text(.035, .95, rf'$\mathrm{{({letter})\ {name}}}$', transform=ax.transAxes,
                    fontsize=10, ha='left', va='top')
            ax.tick_params(top=True, right=True, pad=2.5)
            ax.set_xlim(-.2, .2)
            ax.set_xticks([-.2, -.1, 0., .1, .2])
            ax.set_ylabel(r'$\sigma\;(\sigma(0))$', labelpad=3.)
            ax.legend(loc='lower center' if coupling == 'phase' else 'upper right',
                      frameon=coupling == 'phase', facecolor='white', edgecolor='none',
                      framealpha=1., borderaxespad=.5, borderpad=.1, handlelength=1.7,
                      handletextpad=.4, labelspacing=.15)
        axes[0].set_ylim(.9935, 1.0135)
        axes[0].set_yticks([.995, 1., 1.005, 1.010])
        axes[0].yaxis.set_major_formatter(FormatStrFormatter(r'$%.3f$'))
        axes[1].set_ylim(.99665, 1.0019)
        axes[1].set_yticks([.997, .998, .999, 1., 1.001])
        axes[1].yaxis.set_major_formatter(FormatStrFormatter(r'$%.3f$'))
        axes[2].set_ylim(.85, 1.94)
        axes[2].set_yticks([1., 1.2, 1.4, 1.6, 1.8])
        axes[2].yaxis.set_major_formatter(FormatStrFormatter(r'$%.1f$'))
        axes[-1].set_xlabel(r'$k\;(\tau_r^{-1})$', labelpad=3.)
        weak_handle = Line2D([], [], color='#444444', linewidth=.8,
                             linestyle=(0, (3, 2.3)), label=r'$\mathrm{Weak\ coupling}$')
        fig.legend(handles=[weak_handle], loc='lower center', bbox_to_anchor=(.59, .008),
                   frameon=False, fontsize=8.5, handlelength=2., borderaxespad=0.)
        fig.align_ylabels(axes)
        for extension in ['png', 'svg', 'eps']:
            target = folder/f'figure13.{extension}'
            fig.savefig(target, dpi=600, transparent=False, facecolor='white')
            print(target.resolve(), flush=True)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    compute_parser = commands.add_parser('compute', help='Solve stationary theory in a separate folder.')
    compute_parser.add_argument('--outdir', type=Path, required=True)
    for command in ['validate', 'plot']:
        child = commands.add_parser(command)
        child.add_argument('--outdir', type=Path, default=HERE)
    args = parser.parse_args()
    {'compute': compute, 'validate': validate, 'plot': plot}[args.command](args.outdir)


if __name__ == '__main__':
    main()
