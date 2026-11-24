#!/usr/bin/env python3
"""Figure 6: N fully Cartesian-coupled oscillators.

Plot with saved data or calculate the recorded simulation and quadrature settings.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from numba import set_num_threads
from scipy.stats import t as student_t
from cartesian_n_numerics import (free_second, strong_ratio, strong_contour,
    stationary_ratio, reference_ratio, simulate_replicas)

HERE=Path(__file__).resolve().parent
NUMBERS=(2,4,8,16)
RAW_FIELDS = ('work', 'moment', 'replica_seeds')


def interval(values):
    """Pointwise 95% Student interval across independent complete systems."""
    values=np.asarray(values,dtype=float)
    if len(values)<2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite independent observations')
    mean=float(values.mean());sem=float(values.std(ddof=1)/math.sqrt(len(values)))
    return dict(mean=mean,sem=sem,half=float(student_t.ppf(.975,len(values)-1)*sem),count=len(values))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parameters(root):
    return json.loads((root/'data/parameters.json').read_text())


def run_configs(plan):
    defaults = plan['run_defaults']
    return {run['id']: {**defaults, **{key: value for key, value in run.items() if key != 'id'}}
            for run in plan['runs']}


def verify_checksums(root, checksums):
    for name, digest in checksums.items():
        path = root/'data'/name
        if not path.is_file() or checksum(path) != digest:
            raise ValueError(f'Missing data or checksum mismatch: {name}')


def read_replicas(root, plan, configs):
    """Read per-system EPR estimates and RNG seeds."""
    verify_checksums(root, {'replicas.npz': plan['checksums']['replicas.npz']})
    arrays = {}
    with np.load(root/'data/replicas.npz', allow_pickle=False) as archive:
        for run_id, config in configs.items():
            for field in RAW_FIELDS:
                key = f'{run_id}_{field}'
                values = archive[key]
                if values.shape != (config['replicas'],) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid replica array: {key}')
                arrays[key] = values
            if len(np.unique(arrays[f'{run_id}_replica_seeds'])) != config['replicas']:
                raise ValueError(f'Duplicate seeds in {run_id}')
    return arrays


def compute(root, threads=None, cases=None, *, theory_dir=None):
    """Calculate selected runs from the recorded manuscript parameters."""
    from shutil import copyfile

    root = Path(root)
    plan = load_parameters(HERE)
    configs = run_configs(plan)
    cases = set(cases or ())
    unknown = cases - configs.keys()
    if unknown:
        raise ValueError('Unknown run IDs: ' + ', '.join(sorted(unknown)))
    runs = {run_id: config for run_id, config in configs.items() if not cases or run_id in cases}
    theory_source = theory_dir or HERE
    theory_files = {}
    if not cases:
        names = ('theory.npz', 'numerics.csv')
        if theory_dir is None:
            theory_files = {name: plan['checksums'][name] for name in names}
            verify_checksums(theory_source, theory_files)
        else:
            theory_files = {name: checksum(theory_source/'data'/name) for name in names}
    if threads is not None:
        set_num_threads(threads)

    archive = {}
    for run_id, config in runs.items():
        values = simulate_replicas(**config)
        archive.update({f'{run_id}_{name}': values[name] for name in RAW_FIELDS})
        print(f"Calculated {run_id}: N={config['number']}, k={config['k']:g}", flush=True)
    data = root/'data'
    save_npz(data/'replicas.npz', archive)
    plan['checksums'] = {'replicas.npz': checksum(data/'replicas.npz'), **theory_files}
    if not cases:
        spec = plan['inset']
        traces = {}
        for number in spec['numbers']:
            values = simulate_replicas(**spec['config'], number=number, trace_interval=spec['interval'])
            traces[f'x_{number}'] = values['trace_x'][0]
            traces['time'] = values['trace_time']
        save_npz(data/'sample_trajectories.npz', traces)
        plan['checksums']['sample_trajectories.npz'] = checksum(data/'sample_trajectories.npz')
        for name in theory_files:
            copyfile(theory_source/'data'/name, data/name)
    else:
        plan['completed_runs'] = list(runs)
        print('Selected runs only; this is not a complete figure data set.')
    write_json(data/'parameters.json', plan)


def calculate_theory():
    """Strong-coupling levels for panel (a) and analytical grid for panel (b)."""
    betas, numbers = np.linspace(.1, 10, 100), np.linspace(2, 9.9, 80)
    return dict(beta=betas, number=numbers, ratio=strong_contour(betas, numbers),
                strong_values=np.array([strong_ratio(1., n) for n in NUMBERS]))


def calculate_finite_theory(beta=1., numbers=NUMBERS):
    """Numerically evaluate the exact finite-k stationary formula for panel (a).

    Exchangeability gives sigma/(N*sigma0)=<r_i^2>_k/<r^2>_0. The
    Two-dimensional Gaussian decoupling retains the Cartesian pair interactions
    and angular integration; it is not a strong-coupling approximation.
    """
    couplings = np.unique(np.r_[0.,np.geomspace(.001,.02,12),
                               np.linspace(.02,2.,100),np.linspace(2.,10.5,69),1.,10.])
    settings = dict(order=128,inner_order=96,width=12.)
    numbers = np.asarray(numbers, dtype=int)
    ratios = np.empty((len(numbers), len(couplings)))
    for i, number in enumerate(numbers):
        print(f'Stationary EPR curve: N={number}, beta={beta:g}', flush=True)
        for j, k in enumerate(couplings):
            ratios[i,j] = stationary_ratio(beta, float(k), int(number), **settings)
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ArithmeticError('Nonfinite or nonpositive stationary EPR curve')
    return dict(finite_k=couplings, finite_numbers=numbers, finite_ratio=ratios,
                finite_beta=np.asarray(beta))


def numerical_checks(plan):
    """Convergence of the plotted stationary EPR and its strong-coupling limit."""
    beta = plan['run_defaults']['beta']
    rows = []
    def record(check, number, beta, k, comparison, baseline, refined):
        rows.append(dict(check=check, number=number, beta=beta, k=k,
            comparison=comparison, quantity='normalized_epr', baseline=baseline,
            refined=refined, absolute_difference=abs(refined-baseline)))
    for number in NUMBERS:
        baseline = stationary_ratio(beta, .001, number)
        refined = stationary_ratio(beta, .001, number, order=192, inner_order=144,
                                    rmax=7., width=14.)
        record('quadrature', number, beta, .001, 'increase_nodes_and_extend_domains',
               baseline, refined)
    cases = ((2, 1.), (2, 10.), (4, 1.), (8, plan['minima']['8']['k']),
             (16, plan['minima']['16']['k']), (16, 10.))
    refinements = (('increase_nodes', dict(order=192, inner_order=144)),
                   ('extend_cutoff', dict(rmax=7.)),
                   ('extend_gaussian_width', dict(width=14.)))
    for number, k in cases:
        baseline = stationary_ratio(beta, k, number)
        for comparison, settings in refinements:
            refined = stationary_ratio(beta, k, number, **settings)
            record('quadrature', number, beta, k, comparison, baseline, refined)
        record('quadrature', number, beta, k, 'vs_adaptive', baseline,
               reference_ratio(beta, k, number))
    for number in (2, 16):
        for beta in (.1, 10.):
            baseline, refined = (strong_ratio(beta, number, dps=digits) for digits in (35, 70))
            record('strong_limit', number, beta, '', 'double_decimal_digits', baseline, refined)
    return rows


def write_theory(root, plan):
    """Save stationary curves, simulation references and numerical comparisons."""
    beta = plan['run_defaults']['beta']
    theory = {**calculate_theory(), **calculate_finite_theory(beta)}
    couplings = set(theory['finite_k'])
    points = sorted({(c['number'], c['beta'], c['k']) for c in run_configs(plan).values()
                     if c['number'] not in NUMBERS or c['beta'] != beta or c['k'] not in couplings})
    theory.update(reference_number=np.array([n for n, b, k in points]),
                  reference_beta=np.array([b for n, b, k in points]),
                  reference_k=np.array([k for n, b, k in points]),
                  reference_epr=np.array([stationary_ratio(b, k, n) for n, b, k in points]))
    rows = numerical_checks(plan)
    save_npz(root/'data/theory.npz', theory)
    with (root/'data/numerics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved stationary curves and {len(rows)} numerical comparisons.', flush=True)


ESTIMATORS = {'work': 'stratonovich_epr', 'moment': 'radial_moment_epr'}
PLOT_FIELDS = ('number', 'k', 'stratonovich_epr', 'stratonovich_epr_sem',
               'radial_moment_epr', 'radial_moment_epr_sem')
REPORT_NORMALIZATION = 'EPR estimates and differences are divided by the total uncoupled EPR, N sigma_0.'
REPORT_UNCERTAINTY = (
    'Pointwise 95% Student intervals across independent oscillator populations. '
    'The two EPR estimators share paths and their difference is paired. '
    'Refinement changes are refined minus baseline with Welch intervals for independent runs. '
    'Replica-count estimates use nested subsets.')


def estimate_fields(name, estimate):
    return {name: estimate['mean'], name+'_sem': estimate['sem'],
            name+'_ci95_half_width': estimate['half']}


def welch_change(baseline, refined):
    base_var, refined_var = baseline['sem']**2, refined['sem']**2
    sem = math.sqrt(base_var+refined_var)
    df = (base_var+refined_var)**2 / (base_var**2/(baseline['count']-1)
                                    + refined_var**2/(refined['count']-1)) if sem else 1e30
    return dict(mean=refined['mean']-baseline['mean'], sem=sem,
                half=float(student_t.ppf(.975, df)*sem))


def validate(root):
    """Replay panel (a) estimator benchmarks, refinements and minimum depths."""
    plan = load_parameters(root)
    if 'completed_runs' in plan:
        raise ValueError('Validation requires the complete simulation plan')
    configs = run_configs(plan)
    archive = read_replicas(root, plan, configs)
    verify_checksums(root, {'theory.npz': plan['checksums']['theory.npz']})
    theory = load_npz(root/'data/theory.npz')
    finite_beta = float(theory['finite_beta'])
    quadrature = {(int(number), finite_beta, float(k)): float(theory['finite_ratio'][i, j])
                  for i, number in enumerate(theory['finite_numbers'])
                  for j, k in enumerate(theory['finite_k'])}
    quadrature.update({(int(n), float(b), float(k)): float(value)
                       for n, b, k, value in zip(theory['reference_number'], theory['reference_beta'],
                                                theory['reference_k'], theory['reference_epr'])})

    normalization, estimates, benchmarks = {}, {}, {}
    for run_id, config in configs.items():
        number, beta, k = config['number'], config['beta'], config['k']
        if (number, beta, k) not in quadrature:
            raise ValueError(f'No stationary reference for {run_id}: N={number}, beta={beta}, k={k}')
        norm = number*beta*config['omega']**2*free_second(beta)
        normalization[run_id] = norm
        work, moment = archive[run_id+'_work'], archive[run_id+'_moment']
        estimates[run_id] = {'work': interval(work/norm), 'moment': interval(moment/norm)}
        row = dict(run_id=run_id, number=number, k=k, quadrature_epr=quadrature[number, beta, k])
        for field, name in ESTIMATORS.items():
            row.update(estimate_fields(name, estimates[run_id][field]))
        row.update(estimate_fields('stratonovich_minus_radial_moment', interval((work-moment)/norm)))
        benchmarks[run_id] = row

    refinements = []
    for check in plan['refinements']:
        baseline, refined = check['baseline_run'], check['refined_run']
        if np.intersect1d(archive[baseline+'_replica_seeds'], archive[refined+'_replica_seeds']).size:
            raise ValueError('Refinement comparisons require independent random streams')
        config = configs[baseline]
        row = dict(changed=check['changed'], baseline_run=baseline, refined_run=refined,
                   number=config['number'], k=config['k'])
        for field, name in ESTIMATORS.items():
            change = welch_change(estimates[baseline][field], estimates[refined][field])
            row.update(estimate_fields(name+'_change', change))
        refinements.append(row)

    replica_counts = []
    for check in plan['replica_counts']:
        run_id = check['run_id']
        config = configs[run_id]
        for count in check['replicas']:
            row = dict(run_id=run_id, number=config['number'], k=config['k'], replicas=count)
            for field, name in ESTIMATORS.items():
                estimate = interval(archive[run_id+'_'+field][:count]/normalization[run_id])
                row.update(estimate_fields(name, estimate))
            replica_counts.append(row)

    plotted = [benchmarks[run_id] for run_id in plan['figure_runs']['run_ids']]
    depths = []
    for number in (8, 16):
        k = plan['minima'][str(number)]['k']
        row = next(row for row in plotted if row['number']==number and row['k']==k)
        limit = strong_ratio(finite_beta, number)
        for name in ESTIMATORS.values():
            depths.append(dict(number=number, k=k, estimator=name,
                depth_below_strong_coupling=limit-row[name],
                depth_below_strong_coupling_ci95_half_width=row[name+'_ci95_half_width'],
                quadrature_depth_below_strong_coupling=limit-row['quadrature_epr']))

    validation = dict(normalization=REPORT_NORMALIZATION, uncertainty=REPORT_UNCERTAINTY,
        estimator_benchmarks=list(benchmarks.values()), timestep_burn_duration_refinements=refinements,
        replica_count_checks=replica_counts, minimum_depths=depths)
    figure = {'rows': [{field: row[field] for field in PLOT_FIELDS} for row in plotted]}
    write_json(root/'data/validation.json', validation)
    write_json(root/'data/figure.json', figure)
    print(f'Saved {len(benchmarks)} simulation summaries and {len(refinements)} refinement comparisons.',
          flush=True)


def publication_style(plt):
    return plt.rc_context({'text.usetex': True, 'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'], 'font.size': 9,
        'axes.labelsize': 11.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'legend.fontsize': 9, 'legend.framealpha': 1., 'axes.linewidth': .8,
        'axes.grid': False,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'savefig.bbox': 'tight', 'savefig.transparent': False, 'savefig.facecolor': 'white'})


def save_figure(fig, root, name, outdir=None):
    out = Path(outdir) if outdir is not None else root/'figures'
    out.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'{name}.{extension}', dpi=300)


def plot(root, *, outdir=None):
    """No simulation, theory evaluation, statistics, or quadrature during plotting."""
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'energetics-figure6-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    theory = load_npz(root/'data/theory.npz')
    traces = load_npz(root/'data/sample_trajectories.npz')
    trace_settings = load_parameters(root)['inset']['config']
    figure = json.loads((root/'data/figure.json').read_text())
    colors = ('lightcoral', 'sandybrown', '#a6cee3', '#377eb8')
    with publication_style(plt):
        fig, (ax, contour_ax) = plt.subplots(2, 1, figsize=(4.9, 7.1), layout='constrained')
        for i, number in enumerate(NUMBERS):
            index = int(np.flatnonzero(theory['finite_numbers']==number)[0])
            ax.plot(theory['finite_k'], theory['finite_ratio'][index], color=colors[i],
                    lw=1.6, zorder=3)
            label_k = 8.5
            label_ratio = np.interp(label_k, theory['finite_k'], theory['finite_ratio'][index])
            ax.annotate(rf'$N={number}$', (label_k, label_ratio),
                        xytext=(0, -5 if number == 16 else 5), textcoords='offset points',
                        fontsize=11.5, color='black', ha='center',
                        va='top' if number == 16 else 'bottom')
            rows = sorted((r for r in figure['rows'] if r['number']==number), key=lambda r:r['k'])
            for estimator, marker, offset in (('stratonovich_epr', 'x', -.035),
                                               ('radial_moment_epr', 'o', .035)):
                ax.errorbar([r['k']+offset for r in rows],
                    [r[estimator] for r in rows], yerr=[r[estimator+'_sem'] for r in rows],
                    fmt=marker, mfc='white', ms=3.5, color=colors[i], capsize=2,
                    lw=1.2, elinewidth=.9, linestyle='none')
            ax.axhline(theory['strong_values'][i], color=colors[i], ls='--', lw=1.25)
        ax.set(xlabel=r'$k\;(\tau_r^{-1})$', ylabel=r'$\sigma\;(N\sigma_0)$', xlim=(-.25, 10.5))
        handles = [Line2D([], [], color='black', lw=1.6, label=r'$\mathrm{theory}$'),
                   Line2D([], [], color='.2', marker='o', mfc='white', ls='none',
                          label=r'$\mathrm{radial\ moments}$'),
                   Line2D([], [], color='.2', marker='x', ls='none',
                          label=r'$\mathrm{Stratonovich}$')]
        ax.legend(handles=handles, ncol=1, loc='center', bbox_to_anchor=(.57, .75),
                  fontsize=11.5, frameon=False, handlelength=1.2)
        ax.text(.975, .975, r'$\mathrm{(a)}$', transform=ax.transAxes,
                fontsize=12.5, color='black', ha='right', va='top',
                bbox=dict(facecolor='white', edgecolor='none', pad=1))
        ax.text(.5, .975, rf'$\beta={float(theory["finite_beta"]):g}$',
                transform=ax.transAxes, fontsize=12.5, ha='center', va='top',
                bbox=dict(facecolor='white', edgecolor='none', pad=1))
        grid = theory['ratio']
        # Match the original manuscript contour's linear colour normalization.
        contour = contour_ax.contourf(theory['number'], theory['beta'], grid,
                                      levels=20, cmap='coolwarm')
        contour_ax.set(xlabel=r'$N$', ylabel=r'$\beta$', xlim=(2,9.9), ylim=(.1,10))
        contour_ax.set_xticks([2,4,6,8])
        contour_ax.text(.965, .035, r'$\mathrm{(b)}$', transform=contour_ax.transAxes,
                        fontsize=12.5, color='black', ha='right', va='bottom')
        inset = contour_ax.inset_axes([.10, .665, .64, .24])
        inset.text(.5, 1.02,
                   rf'$\beta={trace_settings["beta"]:g},\quad '
                   rf'k={trace_settings["k"]:g}\;\tau_r^{{-1}}$',
                   transform=inset.transAxes, fontsize=12.5,
                   ha='center', va='bottom')
        for number, color in ((2, 'lightcoral'), (50, '#377eb8')):
            inset.plot(traces['time'], traces['x_'+str(number)], color=color,
                       lw=1.6)
            inset.text(.98 if number == 2 else .02, .96 if number == 2 else .05, rf'$N={number}$',
                       transform=inset.transAxes, fontsize=12.5,
                       color=color, ha='right' if number == 2 else 'left',
                       va='top' if number == 2 else 'bottom')
        inset.set(xlim=(0,20), ylim=(-3.1,3.1), xticks=[0,10,20], yticks=[0],
                  xlabel=r'$t\;(\tau_r)$', ylabel=r'$x\;(r_c)$')
        inset.tick_params(labelsize=10.5, length=3, pad=1)
        inset.xaxis.label.set_size(12.5)
        inset.xaxis.labelpad = 0
        inset.yaxis.label.set_size(12.5)
        inset.yaxis.set_label_position('right')
        inset.yaxis.tick_right()
        colorbar = fig.colorbar(contour, ax=contour_ax, fraction=.055, pad=.025)
        colorbar.set_label(r'$\sigma(k\to\infty)\;(N\sigma_0)$', fontsize=13)
        for panel in (ax, contour_ax):
            panel.xaxis.label.set_size(13)
            panel.yaxis.label.set_size(13)
        save_figure(fig, root, 'figure6', outdir)
        plt.close(fig)
    print(f'Saved PNG and SVG figures in {Path(outdir) if outdir is not None else root/"figures"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="plot", choices=("plot", "compute", "validate", "theory"))
    parser.add_argument("--outdir", type=Path, help="Data directory; compute requires a separate, empty directory")
    parser.add_argument("--threads", type=int, help="Numba threads during simulation")
    parser.add_argument("--case", action="append", help="Compute only this recorded run ID (repeatable)")

    parser.add_argument("--theory-dir", type=Path, help="Use theory and numerical checks from this calculation directory")
    args = parser.parse_args(argv)
    root = (args.outdir or HERE).resolve()
    if args.case and args.command != "compute":
        parser.error("--case requires a computation command")
    if args.theory_dir is not None and (args.command != "compute" or args.case):
        parser.error("--theory-dir requires a full compute run")
    if args.command == "compute":
        if args.outdir is None or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
            parser.error("Choose a separate, empty --outdir for new calculations")
        compute(root, args.threads, args.case, theory_dir=args.theory_dir)
    elif args.command == "validate":
        validate(root)
    elif args.command == "theory":
        if args.outdir is None or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
            parser.error("Choose a separate, empty --outdir for new quadrature")
        write_theory(root, load_parameters(HERE))
    else:
        plot(root)


if __name__=='__main__':
    main()
