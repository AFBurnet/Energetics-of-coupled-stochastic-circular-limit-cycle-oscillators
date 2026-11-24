#!/usr/bin/env python3
"""Figure 5: full Cartesian coupling, saved trajectories and numerical checks."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'energetics-cartesian-matplotlib'))
import numpy as np
from numba import set_num_threads
from scipy.stats import t as student_t
from cartesian_numerics import (sigma0,strong_ratio,theory_moments,radial_marginal,
    radial_domain,reference_moments,find_minimum,
    simulate_replicas,interval,variance_interval)

HERE=Path(__file__).resolve().parent
BETAS=(1.,10/3,7.,10.)
POINT_FIELDS = ('beta', 'k', 'stratonovich_epr', 'stratonovich_epr_sem',
                'radial_moment_epr', 'radial_moment_epr_sem', 'mean_radius', 'mean_radius_sem',
                'radial_variance', 'radial_variance_sem')
RAW_FIELDS = ('work', 'mean', 'second', 'replica_seeds')


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def load_npz(path):
    with np.load(path,allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def save_npz(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**values)
    temporary.replace(path)


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def load_parameters(root):
    return json.loads((root/'data/parameters.json').read_text())


def run_configs(params):
    defaults = params['run_defaults']
    histogram_runs = set(params['histogram']['run_ids'])
    runs = []
    for run in params['runs']:
        run_id = run['id']
        config = dict(defaults, beta=run['beta'], k=run['k'])
        config.update({name: run[name] for name in defaults if name in run})
        runs.append(dict(id=run_id, stage=run.get('stage', 'main'), kind=run.get('kind', ''),
                         base_id=run.get('base_id', ''), config=config, histogram=run_id in histogram_runs))
    return runs


def histogram_edges(params):
    settings = params['histogram']
    return np.linspace(0., settings['max_radius'], settings['bins']+1)


def verify_checksums(root, checksums):
    for name, digest in checksums.items():
        path = root/'data'/name
        if not path.is_file() or checksum(path) != digest:
            raise ValueError(f'Missing data or checksum mismatch: {name}')


def read_data(root):
    root = Path(root)
    params = load_parameters(root)
    if 'completed_runs' in params:
        raise ValueError('Validation requires the complete simulation plan')
    replica_checksums = {name: params['checksums'][name]
                         for name in ('replicas.npz', 'replicas_moments.npz')
                         if name in params['checksums']}
    verify_checksums(root, replica_checksums)
    runs = run_configs(params)
    expected = {f"{run['id']}_{field}" for run in runs for field in RAW_FIELDS}
    expected.update(f"{run['id']}_{field}" for run in runs if run['histogram']
                    for field in ('histogram_counts', 'histogram_samples'))
    archive = {}
    for name in replica_checksums:
        with np.load(root/'data'/name, allow_pickle=False) as values:
            for key in values.files:
                if key in expected:
                    field = values[key]
                    if key in archive and not np.array_equal(archive[key], field):
                        raise ValueError(f'Conflicting replica field: {key}')
                    archive[key] = field
    for run in runs:
        count, run_id = run['config']['replicas'], run['id']
        for field in RAW_FIELDS:
            key = f'{run_id}_{field}'
            values = archive[key]
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f'Invalid replica field: {key}')
        if len(np.unique(archive[f'{run_id}_replica_seeds'])) != count:
            raise ValueError(f'Duplicate replica seeds in {run_id}')
    return params, archive


def compute(root, threads=None, cases=None, theory_dir=None):
    """Calculate selected runs from the recorded manuscript parameters."""
    from shutil import copyfile
    root = Path(root)
    params = load_parameters(HERE)
    plan = run_configs(params)
    cases = set(cases or ())
    unknown = cases - {run['id'] for run in plan}
    if unknown:
        raise ValueError("Unknown run IDs: " + ", ".join(sorted(unknown)))
    runs = [run for run in plan if not cases or run['id'] in cases]
    theory_files = {}
    if not cases:
        names = ('theory.npz', 'numerics.csv')
        if theory_dir is None:
            theory_dir = HERE
            theory_files = {name: params['checksums'][name] for name in names}
            verify_checksums(theory_dir, theory_files)
        else:
            theory_files = {name: checksum(theory_dir/'data'/name) for name in names}
    if threads is not None:
        set_num_threads(threads)
    archive = {}
    for run in runs:
        config = run['config']
        options = (dict(hist_edges=histogram_edges(params),
                        hist_interval=params['histogram']['interval']) if run['histogram'] else {})
        result = simulate_replicas(**config, **options)
        names = list(RAW_FIELDS)
        if run['histogram']:
            names += ['histogram_counts', 'histogram_tails', 'histogram_samples']
        archive.update({run['id']+'_'+name: np.asarray(result[name]) for name in names})
        print(f"Calculated {run['id']}: beta={config['beta']:g}, k={config['k']:g}", flush=True)
    data = root/'data'
    save_npz(data/'replicas.npz', archive)
    params['checksums'] = {'replicas.npz': checksum(data/'replicas.npz'), **theory_files}
    for name in theory_files:
        copyfile(theory_dir/'data'/name, data/name)
    if cases:
        params['completed_runs'] = [run['id'] for run in runs]
        print("Selected runs only; this is not a complete figure data set.")
    write_json(data/'parameters.json', params)


def calculate_theory(settings):
    numerical=dict(order=settings['order'],tail_log=settings['tail_log'])
    ks=np.unique(np.r_[np.linspace(0,2,101),np.linspace(2,10.5,101)])
    records=[[theory_moments(beta,k,**numerical) for k in ks] for beta in BETAS]
    moment_k=np.linspace(0,40,101)
    moments=[theory_moments(7,k,**numerical) for k in moment_k]
    radius=np.linspace(0,3.2,501)
    strong_beta=np.unique(np.r_[0.,np.geomspace(.001,1,80),np.linspace(1,100,251)])
    envelope_beta=np.unique(np.r_[np.geomspace(.1,3,18),np.linspace(3,10,31),np.linspace(10,100,61),10/3,7.])
    minima=[find_minimum(beta,kmax=settings['minimum_kmax'],order=settings['order']) for beta in envelope_beta]
    return dict(beta=np.asarray(BETAS),k=ks,ratio=np.array([[r['ratio'] for r in rs] for rs in records]),
        limits=np.array([strong_ratio(b) for b in BETAS]),moment_k=moment_k,
        mean=np.array([r['mean'] for r in moments]),variance=np.array([r['variance'] for r in moments]),
        radius=radius,marginal=np.array([radial_marginal(7,k,radius,**numerical) for k in (0,40)]),
        strong_beta=strong_beta,strong_ratio=np.array([strong_ratio(b) for b in strong_beta]),
        envelope_beta=envelope_beta,envelope_ratio=np.array([min(r['ratio'],r['strong_ratio']) for r in minima]),
        envelope_strong_ratio=np.array([r['strong_ratio'] for r in minima]),
        envelope_k=np.array([r['k'] for r in minima]),envelope_boundary=np.array([r['boundary'] for r in minima]))


ESTIMATORS = {'work': 'stratonovich_epr', 'moment': 'radial_moment_epr'}
REFINEMENT_NAMES = {'dt_half': 'half_timestep', 'dt_quarter': 'quarter_timestep',
                    'burn': 'burn_in', 'measure': 'measurement_duration'}
REPORT_NORMALIZATION = 'EPR estimates and differences are divided by the uncoupled pair EPR.'
REPORT_UNCERTAINTY = (
    'Pointwise 95% Student intervals across independent oscillator pairs. The two EPR '
    'estimators share paths and their difference is paired. Refinement changes are refined '
    'minus baseline, with Welch intervals for independent runs. The depth from the uncoupled '
    'estimate also uses a Welch interval. Replica-count estimates use nested subsets. '
    'Radial variance uses a replica jackknife.')
BOOTSTRAP_QUANTILES = (.025, .5, .975)


def values_for(run, archive):
    values = {name: archive[run['id']+'_'+name] for name in RAW_FIELDS}
    config = run['config']
    values['moment'] = 2*config['beta']*config['omega']**2*values['second']
    return values


def independent_difference(a, b):
    """Mean(a)-mean(b), with independent-sample Welch uncertainty."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        raise ValueError('Need two independent samples in each group')
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    variance = va+vb
    df = (variance**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)) if variance else len(a)+len(b)-2)
    sem = math.sqrt(variance)
    return (float(a.mean()-b.mean()), sem, float(student_t.ppf(.975, df)*sem))


def estimate_fields(name, estimate):
    mean, sem, half = estimate
    return {name: float(mean), name+'_sem': float(sem), name+'_ci95_half_width': float(half)}


def prepare_summaries(root, params, runs, archive):
    rows, points, histograms = [], [], []
    edges = histogram_edges(params)
    for run in runs:
        config = run['config']
        beta, k = config['beta'], config['k']
        values = values_for(run, archive)
        norm = 2*sigma0(beta, config['omega'])
        row = dict(run_id=run['id'], beta=beta, k=k, replicas=config['replicas'],
                   quadrature_epr=theory_moments(beta, k)['ratio'])
        for field, name in ESTIMATORS.items():
            row.update(estimate_fields(name, interval(values[field]/norm)))
        row.update(estimate_fields('stratonovich_minus_radial_moment',
                                   interval((values['work']-values['moment'])/norm)))
        row.update(estimate_fields('mean_radius', interval(values['mean'])))
        row.update(estimate_fields('radial_variance', variance_interval(values['mean'], values['second'])))
        rows.append(row)
        if run['stage'] in ('main', 'moments'):
            points.append([row[name] for name in POINT_FIELDS])
        if run['histogram']:
            samples = float(archive[run['id']+'_histogram_samples'])
            counts = archive[run['id']+'_histogram_counts']
            density = counts/samples/np.diff(edges)[None, :]
            means = density.mean(axis=0)
            sems = density.std(axis=0, ddof=1)/np.sqrt(len(density))
            histograms.append((k, means, sems))
    histograms.sort(key=lambda row: row[0])
    save_npz(root/'data/figure.npz', dict(
        point_fields=np.asarray(POINT_FIELDS), points=np.asarray(points), histogram_edges=edges,
        histogram_mean=np.array([row[1] for row in histograms]),
        histogram_sem=np.array([row[2] for row in histograms])))
    return rows


def numerical_checks(root,params):
    """Refinements and independent integrals for the quantities in Figure 5."""
    from numpy.polynomial.legendre import leggauss
    from cartesian_numerics import _potential
    order,tail = params['theory']['order'],params['theory']['tail_log']
    rows = []
    def record(check,beta,k,comparison,quantity,baseline,refined,**extra):
        rows.append(dict(check=check,beta=beta,k=k,comparison=comparison,quantity=quantity,
            baseline=float(baseline),refined=float(refined),
            absolute_difference=float(abs(refined-baseline)),**extra))
    cases = [(1.,1.),(10/3,params['minima'][f'{10/3:g}']['k']),
             (7.,params['minima']['7']['k']),(7.,40.),(10.,params['minima']['10']['k'])]
    for beta,k in cases:
        domain = radial_domain(beta,tail)
        base = theory_moments(beta,k,order=order,tail_log=tail)
        comparisons = [
            ('double_nodes',theory_moments(beta,k,order=2*order,tail_log=tail)),
            ('extend_cutoff',theory_moments(beta,k,order=order,rmax=1.25*domain,tail_log=tail))]
        if beta==7:
            comparisons.append(('vs_adaptive',reference_moments(beta,k,rmax=1.25*domain)))
        quantities = [('ratio','normalized_epr')]
        if beta==7: quantities += [('mean','mean_radius'),('variance','radial_variance')]
        for comparison,refined in comparisons:
            for key,quantity in quantities:
                record('quadrature',beta,k,comparison,quantity,base[key],refined[key])
    domain = radial_domain(7,tail)
    nodes,weights = leggauss(2*order)
    radius,weights = domain*(nodes+1)/2,domain*weights/2
    for k in (0.,40.):
        base = radial_marginal(7,k,radius,order=order,rmax=domain,tail_log=tail)
        fine = radial_marginal(7,k,radius,order=2*order,rmax=1.25*domain,tail_log=tail)
        record('marginal',7,k,'vs_unit_mass','density_integral',1,np.dot(weights,base))
        i = np.argmax(abs(fine-base))
        record('marginal',7,k,'double_nodes_and_extend_cutoff','radial_density',base[i],fine[i],radius=radius[i])
    for beta in (.01,1.,10/3,7.,10.,100.):
        domain = 1.25*radial_domain(beta,tail)
        nodes,weights = leggauss(3*order)
        radius,weights = domain*(nodes+1)/2,domain*weights/2
        moments = []
        for inverse_temperature in (beta,2*beta):
            density = weights*radius*np.exp(inverse_temperature*(_potential(radius)-1/6))
            moments.append(np.dot(density,radius**2)/density.sum())
        record('strong_limit',beta,'','vs_fixed_nodes','normalized_epr',strong_ratio(beta),moments[1]/moments[0])
    for beta in (10/3,7.,10.,100.):
        base = find_minimum(beta,order=order)
        fine = find_minimum(beta,order=2*order)
        record('minimum_envelope',beta,base['k'],'double_nodes','normalized_epr',base['ratio'],fine['ratio'])
        record('minimum_envelope',beta,base['k'],'double_nodes','minimizing_coupling',base['k'],fine['k'])
    for beta in (1.,2.5,3.):
        base = find_minimum(beta,kmax=400.,order=order)
        fine = find_minimum(beta,kmax=4000.,order=order)
        record('minimum_envelope',beta,base['k'],'extend_search','normalized_epr',
               min(base['ratio'],base['strong_ratio']),min(fine['ratio'],fine['strong_ratio']),
               k_role='search_boundary' if base['boundary'] else 'finite_search_minimizer')
    write_csv(root/'data/numerics.csv',rows)
    return rows


def write_theory(root, params):
    """Recalculate the stationary curves and numerical checks into a new run."""
    theory = calculate_theory(params['theory'])
    save_npz(root/'data/theory.npz', theory)
    rows = numerical_checks(root, params)
    print(f'Saved stationary curves and {len(rows)} numerical comparisons.', flush=True)


def validate(root):
    root = Path(root)
    params, archive = read_data(root)
    runs = run_configs(params)
    benchmarks = prepare_summaries(root, params, runs, archive)
    indexed = {run['id']: run for run in runs}
    refinements, replica_counts = [], []
    for run in runs:
        if run['stage'] != 'validation':
            continue
        base = indexed[run['base_id']]
        refined, baseline = values_for(run, archive), values_for(base, archive)
        config = run['config']
        norm = 2*sigma0(config['beta'], config['omega'])
        if np.intersect1d(refined['replica_seeds'], baseline['replica_seeds']).size:
            raise ValueError('Validation groups must use independent random streams')
        row = dict(changed=REFINEMENT_NAMES[run['kind']], baseline_run=base['id'], refined_run=run['id'],
                   beta=config['beta'], k=config['k'])
        for field, name in ESTIMATORS.items():
            change = independent_difference(refined[field]/norm, baseline[field]/norm)
            row.update(estimate_fields(name+'_change', change))
        refinements.append(row)
        if run['kind'] == 'dt_half':
            n = len(baseline['work'])
            for count in sorted({max(2, n//4), max(2, n//2)} - {n}):
                row = dict(run_id=base['id'], beta=config['beta'], k=config['k'], replicas=count)
                for field, name in ESTIMATORS.items():
                    row.update(estimate_fields(name, interval(baseline[field][:count]/norm)))
                replica_counts.append(row)

    depths, locations = [], []
    for beta in BETAS[1:]:
        kstar = params['minima'][f'{beta:g}']['k']
        eligible = [run for run in runs if run['stage']=='main' and run['config']['beta']==beta]
        at_min = next(run for run in eligible if run['config']['k']==kstar)
        at_zero = next(run for run in eligible if run['config']['k']==0.)
        minimum, uncoupled = values_for(at_min, archive), values_for(at_zero, archive)
        norm = 2*sigma0(beta, params['run_defaults']['omega'])
        for field, name in ESTIMATORS.items():
            depth = independent_difference(uncoupled[field]/norm, minimum[field]/norm)
            row = dict(beta=beta, k=kstar, estimator=name)
            row.update(estimate_fields('depth_from_uncoupled', depth))
            row.update(depth_below_strong_coupling=strong_ratio(beta)-float(minimum[field].mean()/norm),
                       depth_below_strong_coupling_ci95_half_width=float(interval(minimum[field]/norm)[2]))
            depths.append(row)
        rng = np.random.default_rng(20261028)
        grid = np.array([run['config']['k'] for run in eligible])
        inputs = [values_for(run, archive) for run in eligible]
        draws = {field: [] for field in ESTIMATORS}
        for _ in range(400):
            means = {field: [] for field in ESTIMATORS}
            for values in inputs:
                count = len(values['work'])
                indices = rng.integers(count, size=count)
                for field in ESTIMATORS:
                    means[field].append(values[field][indices].mean())
            for field in ESTIMATORS:
                draws[field].append(float(grid[np.argmin(means[field])]))
        locations.append(dict(beta=beta, sampled_k=grid.tolist(), **{
            name+'_location_quantiles': np.quantile(draws[field], BOOTSTRAP_QUANTILES).tolist()
            for field, name in ESTIMATORS.items()}))

    output = dict(normalization=REPORT_NORMALIZATION, uncertainty=REPORT_UNCERTAINTY,
        estimator_benchmarks=benchmarks, timestep_burn_duration_refinements=refinements,
        replica_count_checks=replica_counts, minimum_depths=depths,
        sampled_minimizers=dict(quantiles=list(BOOTSTRAP_QUANTILES),
            method='Bootstrap minimizers over the sampled coupling grid.', results=locations))
    write_json(root/'data/validation.json', output)
    print(f'Saved {len(benchmarks)} trajectory summaries and {len(refinements)} refinement comparisons.',
          flush=True)


def plot(root, *, outdir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    outdir = Path(outdir) if outdir is not None else root/'figures'
    figure = load_npz(root/'data/figure.npz')
    theory = load_npz(root/'data/theory.npz')
    with publication_style(plt):
        plot_main(figure, theory, outdir, plt)
    print(f'Saved PNG/SVG figures in {outdir}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="plot", choices=("plot", "compute", "validate", "theory"))
    parser.add_argument("--outdir", type=Path, help="Data directory; compute requires a separate, empty directory")
    parser.add_argument("--threads", type=int, help="Numba threads during simulation")
    parser.add_argument("--theory-dir", type=Path, help="Run directory from theory, used by a full compute run")
    parser.add_argument("--case", action="append", help="Compute only this recorded run ID (repeatable)")

    args = parser.parse_args(argv)
    root = (args.outdir or HERE).resolve()
    if args.case and args.command != "compute":
        parser.error("--case requires a computation command")
    if args.theory_dir is not None and (args.command != "compute" or args.case):
        parser.error("--theory-dir requires a full compute run")
    if args.command == "compute":
        if args.outdir is None or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
            parser.error("Choose a separate, empty --outdir for new calculations")
        compute(root, args.threads, args.case, args.theory_dir)
    elif args.command == "validate":
        validate(root)
    elif args.command == "theory":
        if args.outdir is None or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
            parser.error("Choose a separate, empty --outdir for new quadrature")
        write_theory(root, load_parameters(HERE))
    else:
        plot(root)


def publication_style(plt):
    """LaTeX typography for the saved-data Figure 5 plots."""
    return plt.rc_context({
        'text.usetex': True, 'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'], 'font.size': 11,
        'axes.labelsize': 14, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 9, 'legend.framealpha': 1,
        'axes.linewidth': .8, 'axes.grid': False,
        'xtick.major.size': 4.5, 'ytick.major.size': 4.5,
        'xtick.major.width': .8, 'ytick.major.width': .8,
        'savefig.bbox': 'tight', 'savefig.transparent': False,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def plot_main(figure, theory, outdir, plt):
    """Render Figure 5 from saved arrays and standard errors."""
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fields = [str(name) for name in figure['point_fields']]
    points = [dict(zip(fields, row)) for row in figure['points']]
    betas = np.asarray(theory['beta'], dtype=float)
    colors = ('#9fc1d6', '#6b9cbd', '#397aa8', '#15537e')
    pinks = ('#fbd0da', '#f3abc1', '#dc719e', '#ad3f76')
    scale = .64
    legend_size, middle_label_size = 24*scale, 28*scale
    fig = plt.figure(figsize=(9.6*scale, 16.5*scale), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=(1.7, 1., 1.7),
                           hspace=.04, wspace=.06)
    epr_ax = fig.add_subplot(grid[0, :])
    density_ax = fig.add_subplot(grid[1, 0])
    mean_ax = fig.add_subplot(grid[1, 1])
    limit_ax = fig.add_subplot(grid[2, :])
    for axis in (epr_ax, limit_ax):
        axis.set_box_aspect(.70)
    for axis in (density_ax, mean_ax):
        axis.set_box_aspect(1.)

    ratios = np.asarray(theory['ratio'], dtype=float)

    def epr_point(k, value, sem, color, marker):
        if not (np.isfinite(value) and np.isfinite(sem)):
            return
        epr_ax.errorbar(k, value, yerr=abs(float(sem)), fmt=marker, ms=4.1,
                       color=color, mfc='white' if marker == 'o' else color,
                       mew=.95, capsize=2.1, lw=.8, zorder=4 if marker == 'o' else 3)

    for index, (beta, color) in enumerate(zip(betas, colors)):
        label = r'$\beta=10/3$' if np.isclose(beta, 10/3) else rf'$\beta={beta:g}$'
        limit=theory['limits'][index]
        source_k=np.asarray(theory['k']);source_y=ratios[index]
        crossings=np.flatnonzero((source_y[:-1]-limit)*(source_y[1:]-limit)<0)
        roots=[source_k[j]+(limit-source_y[j])*(source_k[j+1]-source_k[j])/
               (source_y[j+1]-source_y[j]) for j in crossings]
        curve_k=np.unique(np.r_[source_k,roots]);curve_y=np.interp(curve_k,source_k,source_y)
        epr_ax.plot(curve_k,np.where(curve_y>=limit-1e-14,curve_y,np.nan),color=color,lw=1.8,zorder=6)
        epr_ax.plot(curve_k,np.where(curve_y<=limit+1e-14,curve_y,np.nan),color=pinks[index],lw=1.8,zorder=6)
        label_k = 8.7
        epr_ax.annotate(label, (label_k, np.interp(label_k, source_k, source_y)),
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=legend_size, color='black', ha='center',
                        va='bottom')
        epr_ax.axhline(limit, color=color, ls='--', lw=.9)
        for row in points:
            if np.isclose(row['beta'], beta) and row['k'] <= 10.5:
                epr_point(row['k'], row['stratonovich_epr'], row['stratonovich_epr_sem'], color, 'x')
                epr_point(row['k'], row['radial_moment_epr'], row['radial_moment_epr_sem'], color, 'o')
    epr_ax.axhline(1, color='.65', lw=.7, zorder=0)
    epr_ax.set(xlabel=r'$k\;(\tau_r^{-1})$', ylabel=r'$\sigma\;(2\sigma_0)$',
               xlim=(-.15, 10.5))
    handles = [Line2D([], [], color='black', lw=1.8, label=r'$\mathrm{theory}$'),
               Line2D([], [], ls='', marker='o', ms=4.5, mfc='white', color='.25',
                       label=r'$\mathrm{radial\ moments}$'),
               Line2D([], [], ls='', marker='x', ms=4.5, color='.25',
                       label=r'$\mathrm{Stratonovich}$')]
    epr_ax.legend(handles=handles, frameon=False, ncol=1,
                  loc='center', bbox_to_anchor=(.45, .80),
                  fontsize=legend_size, handlelength=1.2,
                  labelspacing=.15, borderpad=.1)
    epr_ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    radius = np.asarray(theory['radius'])
    edges = np.asarray(figure['histogram_edges'])
    for index, k in enumerate((0, 40)):
        density_label = (r'$k=0$'+'\n'+r'$r p_{0,\beta}$' if k == 0
                         else r'$k=40$'+'\n'+r'$\approx r p_{0,2\beta}$')
        density_ax.plot(radius, theory['marginal'][index], color='.15', lw=1.8,
                        ls=':' if k == 0 else '-', label=density_label, zorder=3)
        sample_mean = np.asarray(figure['histogram_mean'][index])
        sample_sem = np.asarray(figure['histogram_sem'][index])
        selected_edges = edges if edges.ndim == 1 else edges[index]
        centers = (selected_edges[:-1]+selected_edges[1:])/2
        hist_color = '#c3cbd2' if k == 0 else '#8c9eab'
        if np.any(np.isfinite(sample_mean)):
            density_ax.fill_between(centers, sample_mean-sample_sem,
                                    sample_mean+sample_sem, color=hist_color,
                                    alpha=1., lw=0, zorder=1)
            density_ax.plot(centers, sample_mean, '.', color=hist_color,
                            ms=2.5, zorder=2)
    density_ax.set(xlabel=r'$r\;(r_c)$', ylabel=r'$r p(r)$', xlim=(0, 3))
    density_ax.legend(frameon=False, loc='upper right', fontsize=legend_size,
                      handlelength=.6, handletextpad=.25, borderpad=.1,
                      borderaxespad=.15)
    density_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    density_ax.set_xticks((0, 1, 2, 3))

    variance_ax = mean_ax.twinx()
    variance_ax.set_box_aspect(1.)
    mean_color, variance_color = 'lightcoral', 'lightblue'
    mean_axis_color, variance_axis_color = '#c03029', '#527d9a'
    mean_ax.plot(theory['moment_k'], theory['mean'], color=mean_color, lw=1.9, zorder=6)
    variance_ax.plot(theory['moment_k'], theory['variance'], color=variance_color, lw=1.9, zorder=6)
    moment_points = (0., .25, 1., 4., 10., 20., 30., 40.)
    for row in points:
        if not np.isclose(row['beta'], 7) or not np.any(np.isclose(row['k'], moment_points)):
            continue
        mean_ax.errorbar(row['k'], row['mean_radius'], yerr=row['mean_radius_sem'],
                         fmt='o', color=mean_color, mfc='white', mew=.9,
                         ms=3.8, capsize=2., lw=.8, zorder=4)
        variance_ax.errorbar(row['k'], row['radial_variance'], yerr=row['radial_variance_sem'],
                             fmt='s', color=variance_color, mfc='white', mew=.9,
                             ms=3.6, capsize=2., lw=.8, zorder=4)
    mean_ax.set(xlabel=r'$k\;(\tau_r^{-1})$', ylabel='', xlim=(-1, 41))
    mean_ax.text(.5, .975, r'$\beta=7$', transform=mean_ax.transAxes,
                 fontsize=legend_size, color='black', ha='center', va='top')
    variance_ax.set_ylabel('')
    mean_ax.tick_params(axis='y', colors=mean_axis_color)
    variance_ax.tick_params(axis='y', colors=variance_axis_color)
    for panel in (mean_ax, variance_ax):
        panel.spines['left'].set_color(mean_axis_color)
        panel.spines['right'].set_color(variance_axis_color)
    label_k = 23.
    mean_ax.annotate(r'$\langle r\rangle_k$',
                     (label_k, np.interp(label_k, theory['moment_k'], theory['mean'])),
                     xytext=(0, 8), textcoords='offset points', color=mean_axis_color,
                     fontsize=middle_label_size, ha='center', va='bottom')
    variance_ax.annotate(r'$\mathrm{Var}_k(r)$',
                         (label_k, np.interp(label_k, theory['moment_k'], theory['variance'])),
                         xytext=(0, 7), textcoords='offset points', color=variance_axis_color,
                         fontsize=middle_label_size, ha='center', va='bottom')
    mean_ax.set_xticks((0, 20, 40))
    mean_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    variance_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    beta_grid = np.asarray(theory['strong_beta'], dtype=float)
    limit = np.asarray(theory['strong_ratio'], dtype=float)
    limit_ax.fill_between(beta_grid, limit, 1, where=limit < 1,
                          color='lightblue', lw=0, interpolate=True)
    limit_ax.plot(beta_grid, limit, color='.1', lw=1.8,
                  label=r'$\sigma(k\to\infty)$', zorder=4)
    envelope_beta = np.asarray(theory['envelope_beta'], dtype=float)
    envelope = np.asarray(theory['envelope_ratio'], dtype=float)
    upper = np.asarray(theory['envelope_strong_ratio'])
    valid = np.isfinite(envelope) & (envelope < upper)
    limit_ax.fill_between(envelope_beta, envelope, upper, where=valid,
                          color='#edb3c2', lw=0, interpolate=True)
    limit_ax.plot(envelope_beta, envelope, color='#ac5270', lw=1.1,
                  label=r'$\min_k\sigma(k)$', zorder=3)
    limit_ax.axhline(1, color='.5', ls='--', lw=.8)
    limit_ax.set(xlabel=r'$\beta$', ylabel=r'$\sigma\;(2\sigma_0)$', xlim=(0, 100))
    limit_ax.set_ylim(top=1.045)
    limit_ax.legend(frameon=False, loc='lower right', fontsize=legend_size,
                    labelspacing=.3)
    limit_ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for panel in (epr_ax, density_ax, mean_ax, variance_ax, limit_ax):
        panel.tick_params(labelsize=22*scale, length=7.5*scale, width=1.2)
        panel.xaxis.label.set_size(26*scale)
        panel.yaxis.label.set_size(26*scale)
    for panel, letter, left in ((epr_ax, 'a', .025), (density_ax, 'b', .04),
                                (mean_ax, 'c', .10), (limit_ax, 'd', .025)):
        panel.text(left, .99 if panel is epr_ax else .975,
                   rf'$\mathrm{{({letter})}}$', transform=panel.transAxes,
                   fontsize=legend_size, fontweight='normal', color='black',
                   ha='left', va='top', zorder=10)
    fig.canvas.draw()
    fig.set_layout_engine('none')
    density_ax.set_position(density_ax.get_position().translated(.02, 0))
    mean_position = mean_ax.get_position().translated(-.02, 0)
    mean_ax.set_position(mean_position)
    variance_ax.set_position(mean_position)
    save_figure(fig, outdir, 'figure5')
    plt.close(fig)


def save_figure(fig, outdir, name):
    """Save opaque publication PNG/SVG outputs, without generating PDFs."""
    from pathlib import Path
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(outdir/f'{name}.{suffix}', dpi=300,
                    facecolor='white', transparent=False)

if __name__=='__main__':
    main()
