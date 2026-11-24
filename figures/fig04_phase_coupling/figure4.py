#!/usr/bin/env python3
"""Phase-coupling Figure 4: computation, validation and saved-data plotting."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'energetics-phase-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numba import set_num_threads
from scipy.stats import t as student_t
from numerics import (MEASURES, simulate, solve_pde, solve_pde_contributions,
                      decompose_pde_currents, radial_mesh, radial_weight,
                      free_second, mean_variance_decomposition,
                      trajectory_mean_variance_decomposition, weak_pair_response,
                      radial_extent, bimodal_coefficient, weak_zero_detuning)

HERE = Path(__file__).resolve().parent


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parameters(root):
    return json.loads((root/'data/parameters.json').read_text())


def simulation_records(params):
    model, couplings = params['model'], params['simulation_k']
    defaults = dict(params['simulation'], T_eff=model['T_eff'], omega_bar=model['omega_bar'])
    return [dict(id=f'r{j*len(couplings)+i:03d}', delta_index=j, k_index=i,
                 config=dict(defaults, delta_omega=delta, k=k))
            for j, delta in enumerate(model['deltas']) for i, k in enumerate(couplings)]


def validation_runs(params):
    baselines = {run['id']: run['config'] for run in simulation_records(params)}
    refinements = params['refinements']
    variants = []
    for base_id in refinements['baseline_runs']:
        for kind, overrides in refinements['overrides'].items():
            variants.append(dict(id=f'v{len(variants):03d}', kind=kind, base_id=base_id,
                                 config={**baselines[base_id], **overrides}))
    return variants


def theory_records(params):
    model = params['model']
    mesh = dict(params['mesh'], T_eff=model['T_eff'], omega_bar=model['omega_bar'])
    grids = [('main', {'delta_index': j}, dict(mesh, delta_omega=delta))
             for j, delta in enumerate(model['deltas'])]
    grids.insert(params['inset']['theory_index'], ('inset', {}, {**mesh, **params['inset']['settings']}))
    return [dict(kind=kind, **index, k_index=i, config=dict(settings, k=k))
            for kind, index, settings in grids for i, k in enumerate(params['theory_k'])]


def load_bundle(root):
    root = Path(root)
    params = load_parameters(root)
    missing = [name for name in ('replicas.npz', 'validation_replicas.npz')
               if not (root/'data'/name).is_file()]
    if missing:
        raise FileNotFoundError('Missing raw measurements: '+', '.join(missing)+
                                '. Restore the validation archive as described in data-bundles/README.md.')
    for name, digest in params['checksums'].items():
        path = root/'data'/name
        if not path.exists() or checksum(path) != digest:
            raise ValueError(f'Archive checksum mismatch: {name}')
    replicas, theory = [load_npz(root/'data'/name) for name in ('replicas.npz','theory.npz')]
    for run in simulation_records(params):
        identifier, count = run['id'], run['config']['n_replicas']
        values, seeds = replicas[identifier+'_values'], replicas[identifier+'_seeds']
        if values.shape != (count,len(MEASURES)) or seeds.shape != (count,):
            raise ValueError(f'Invalid independent-replica record {identifier}')
        if not np.isfinite(values).all() or len(np.unique(seeds)) != count:
            raise ValueError(f'Nonfinite observations or duplicate seeds in {identifier}')
    return params, replicas, theory


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite independent replicas')
    mean = float(values.mean())
    sem = float(values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=mean, sem=sem, half95=float(student_t.ppf(.975,len(values)-1)*sem),
                count=len(values))


def figure_arrays(params, replicas, theory):
    """Build the original points, one-SEM bars, and jackknife inset."""
    model = params['model']
    T, omega = model['T_eff'], model['omega_bar']
    m2 = free_second(T)
    deltas = np.asarray(model['deltas'])
    nk = len(params['simulation_k'])
    inset_delta = params['inset']['settings']['delta_omega']
    means, sems = np.empty((len(deltas),nk,7)), np.empty((len(deltas),nk,7))
    inset_values = []
    for run in simulation_records(params):
        j, i = run['delta_index'], run['k_index']
        values = replicas[run['id']+'_values']
        means[j,i] = values.mean(axis=0)
        sems[j,i] = values.std(axis=0,ddof=1)/np.sqrt(len(values))
        if run['config']['delta_omega']==inset_delta:
            inset_values.append((i,trajectory_mean_variance_decomposition(
                values,T,omega,inset_delta,m2)))
    inset_values = [value for _,value in sorted(inset_values)]
    r, mu, _ = radial_mesh(T,params['mesh']['nr'],params['mesh']['rmax'])
    mesh_m2 = float(mu@(r*r))
    inset_theory = []
    for i in range(len(params['theory_k'])):
        record = {key: theory['inset_'+key][i] for key in ('epr','terms','current_terms')}
        inset_theory.append(mean_variance_decomposition(record,T,omega,inset_delta,mesh_m2))
    figure = dict(main_k=np.asarray(params['simulation_k']), theory_k=np.asarray(params['theory_k']),
        deltas=deltas, T_eff=np.asarray(T), omega_bar=np.asarray(omega), m2=np.asarray(m2),
        baseline=(2*omega**2+deltas**2/2)*m2/T,
        main_mean=means, main_sem=sems, pde_epr=theory['main_epr'],
        inset_theory_epr=theory['inset_epr'],
        inset_theory_psi=np.array([r['sigma_psi'] for r in inset_theory]),
        inset_theory_mean=np.array([r['mean_differential_epr'] for r in inset_theory]),
        inset_theory_angular=np.array([r['angular_variation_epr'] for r in inset_theory]),
        inset_theory_radial=np.array([r['radial_variation_epr'] for r in inset_theory]))
    for name,key in (('epr','epr'),('epr_sem','epr_sem'),('psi','sigma_psi'),
                     ('mean','mean_differential_epr'),('mean_sem','mean_differential_sem')):
        figure['inset_sim_'+name] = np.array([r[key] for r in inset_values])
    return figure


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def fresh_output(root):
    root = root.resolve()
    if root == HERE or HERE in root.parents or root.exists() and any(root.iterdir()):
        raise ValueError('Choose an empty output directory outside the bundled figure.')
    root.mkdir(parents=True, exist_ok=True)
    return root


def calculate_boundary(root):
    """Independently solve the C2=0 curve using the recorded settings."""
    from scipy.optimize import brentq
    params = load_parameters(HERE)
    config = params['overlay']['settings']
    nr, upper, tail = config['nr'], config['upper'], config['tail_log']
    entry = brentq(lambda beta: float(weak_pair_response(
        1/beta, [upper], nr, radial_extent(1/beta, tail_log=tail))['C2'][0]),
        config['beta_min'], config['beta_max'], xtol=config['xtol'])
    betas = np.geomspace(entry, config['beta_max'], config['points'])
    rows = [weak_zero_detuning(T=1/float(beta), nr=nr,
             rmax=radial_extent(1/float(beta), tail_log=tail),
             upper=upper, xtol=config['xtol']) for beta in betas]
    data = {key: np.array([r[key] for r in rows]) for key in
            ('detuning', 'C2', 'rmax', 'relative_residual', 'radial_tail_mass')}
    save_npz(root/'data/overlay.npz', dict(data, beta=betas))
    print(f'C2=0 boundary enters at beta={entry:.10f}.', flush=True)


def estimate_fields(name, estimate):
    return {name: estimate['mean'], name+'_sem': estimate['sem'],
            name+'_ci95_half_width': estimate['half95']}


ESTIMATOR_COLUMNS = (('stratonovich_epr', 0), ('observable_epr', 1))
REFINEMENT_NAMES = {'dt_coarser': 'coarser_timestep', 'dt_finer': 'finer_timestep',
                    'burn': 'burn_in', 'duration': 'measurement_duration'}
REPORT_NORMALIZATION = (
    'EPR estimates and differences are divided by the uncoupled pair EPR at the same detuning. '
    'Phase decomposition uses unnormalized EPR.')
REPORT_UNCERTAINTY = (
    'Pointwise 95% Student intervals across independent oscillator pairs. '
    'The two EPR estimators share paths. Refinement changes are variant minus baseline, paired '
    'by replica seed. Replica-count estimates use nested subsets. The phase decomposition '
    'reports jackknife standard errors.')


def validate(root, report=None):
    """Rebuild figure arrays and replay the trajectory and stationary checks."""
    params, replicas, theory = load_bundle(root)
    prepared = figure_arrays(params, replicas, theory)
    bundled = load_npz(root/'data/figure.npz')
    for key, value in prepared.items():
        if not np.allclose(value, bundled[key], rtol=1e-11, atol=1e-13):
            raise ValueError(f'Saved figure disagrees with its numerical inputs: {key}')
    with (root/'data/numerics.csv').open(newline='') as stream:
        references = {f"{float(r['delta_omega']):g}_{float(r['k']):g}": float(r['refined'])
                      for r in csv.DictReader(stream) if r['check']=='pde_domain'}

    T, omega = params['model']['T_eff'], params['model']['omega_bar']
    m2 = free_second(T)
    deltas = params['model']['deltas']
    uncoupled = {delta: (2*omega**2+delta**2/2)*m2/T for delta in deltas}
    reference_m2 = (quad(lambda r: r**3*radial_weight(r,T), 0, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
                    / quad(lambda r: r*radial_weight(r,T), 0, np.inf, epsabs=1e-12, epsrel=1e-12)[0])
    references.update({f'{delta:g}_0': (2*omega**2+delta**2/2)*reference_m2/T for delta in deltas})
    theory_k = np.asarray(params['theory_k'])
    baselines = list(simulation_records(params))
    variants = list(validation_runs(params))
    variant_archive = load_npz(root/'data/validation_replicas.npz')
    estimates, comparisons = [], []

    def benchmark(run, values):
        config = run['config']
        delta, k = config['delta_omega'], config['k']
        norm = uncoupled[delta]
        j = deltas.index(delta)
        point = int(np.argmin(abs(theory_k-k)))
        pde_epr = references.get(f'{delta:g}_{k:g}', float(theory['main_epr'][j, point]))
        row = dict(run_id=run['id'], delta_omega=delta, k=k, replicas=len(values),
                   stationary_pde_epr=pde_epr/norm)
        for name, column in ESTIMATOR_COLUMNS:
            row.update(estimate_fields(name, interval(values[:, column]/norm)))
        row.update(estimate_fields('stratonovich_minus_observable',
                                   interval((values[:, 0]-values[:, 1])/norm)))
        decomposition = trajectory_mean_variance_decomposition(values, T, omega, delta, m2)
        row['phase_decomposition'] = {
            'mean_differential_epr': float(decomposition['mean_differential_epr']),
            'mean_differential_epr_sem': float(decomposition['mean_differential_sem']),
            'current_variation_epr': float(decomposition['variation_epr']),
            'current_variation_epr_sem': float(decomposition['variation_sem']),
        }
        return row

    for run in baselines:
        estimates.append(benchmark(run, replicas[run['id']+'_values']))
    for run in variants:
        run_id, baseline_id = run['id'], run['base_id']
        delta, k = run['config']['delta_omega'], run['config']['k']
        values = variant_archive[run_id+'_values']
        estimates.append(benchmark(run, values))
        baseline = replicas[baseline_id+'_values']
        if not np.array_equal(variant_archive[run_id+'_seeds'], replicas[baseline_id+'_seeds']):
            raise ValueError('Paired validation runs have different replica seeds')
        row = dict(changed=REFINEMENT_NAMES[run['kind']], baseline_run=baseline_id,
                   variant_run=run_id, delta_omega=delta, k=k)
        for name, column in ESTIMATOR_COLUMNS:
            difference = (values[:, column]-baseline[:, column])/uncoupled[delta]
            row.update(estimate_fields(name+'_change', interval(difference)))
        comparisons.append(row)

    data = dict(normalization=REPORT_NORMALIZATION, uncertainty=REPORT_UNCERTAINTY,
        uncoupled_pair_epr_by_detuning={f'{delta:g}': norm for delta, norm in uncoupled.items()},
        collective_phase_epr=2*m2*omega**2/T,
        estimator_benchmarks=estimates, timestep_burn_duration_refinements=comparisons,
        replica_count_checks=replica_count_checks(baselines, variants, replicas, uncoupled))
    if report is not None:
        if report.exists() or HERE in report.resolve().parents:
            raise FileExistsError('Choose an unused report path outside the bundled figure.')
        write_json(report, data)
    print(f'Recomputed {len(estimates)} trajectory summaries and {len(comparisons)} refinement comparisons.',
          flush=True)
    return data


def plot(root):
    figure = load_npz(root/'data/figure.npz')
    minimum = load_npz(root/'data/minimum_grid.npz')
    bimodal = load_npz(root/'data/bimodal.npz')
    settings = json.loads(str(bimodal['settings']))
    overlay = load_npz(root/'data/overlay.npz')
    fig = plot_main(figure, minimum, bimodal, settings, root/'figures', overlay)
    plt.close(fig)
    print(f'Saved PNG/SVG figures in {root/"figures"}', flush=True)


def minimum_map_settings(minimum):
    names = ('beta', 'detuning', 'k', 'omega_bar', 'dt', 'burn_time',
             'measure_time', 'n_trajectories', 'seed')
    return {name: minimum[name].tolist() for name in names}


def heatmap(root, rows=None):
    """Calculate complete beta rows with the published deterministic seed schedule."""
    from phase_epr_simulation_heatmap import simulate_row
    minimum = load_npz(HERE/'data/minimum_grid.npz')
    settings = minimum_map_settings(minimum)
    settings_hash = config_hash(settings)
    rows = range(len(minimum['beta'])) if rows is None else rows
    jobs = len(minimum['detuning'])*len(minimum['k'])*settings['n_trajectories']
    for index in rows:
        beta = float(minimum['beta'][index])
        seed = settings['seed']+index*jobs
        samples = simulate_row(beta, minimum['detuning'], minimum['k'], settings['omega_bar'],
                   settings['dt'], round(settings['burn_time']/settings['dt']),
                   round(settings['measure_time']/settings['dt']), settings['n_trajectories'], seed)
        if not np.isfinite(samples).all():
            raise FloatingPointError(f'Nonfinite values in heatmap row {index}; no samples discarded.')
        save_npz(root/f'data/minimum_rows/beta{index:03d}.npz', dict(samples=samples, beta=beta,
                 seed_start=seed, settings_sha256=settings_hash))
        print(f'Saved heatmap row {index} (beta={beta:g}).', flush=True)
    if rows == range(len(minimum['beta'])):
        assemble_heatmap(root, [root])


def assemble_heatmap(root, sources):
    """Assemble independently computed rows or the exact archived raw rows."""
    minimum = load_npz(HERE/'data/minimum_grid.npz')
    settings_hash = config_hash(minimum_map_settings(minimum))
    repository = HERE.parents[1]
    manifest = json.loads((repository/'data-bundles/validation-data.json').read_text())
    destination = (HERE/'data/minimum_rows').relative_to(repository)
    expected = {}
    for record in manifest['files']:
        path = Path(record.get('restore_path', record['path']))
        if path.parent == destination:
            expected[path.name] = record['sha256']
    n = int(minimum['n_trajectories'])
    jobs = len(minimum['detuning'])*len(minimum['k'])*n
    sigma, sem = [], []
    for index, beta in enumerate(minimum['beta']):
        name = f'data/minimum_rows/beta{index:03d}.npz'
        files = [folder/name for folder in sources if (folder/name).exists()]
        if len(files) != 1:
            raise ValueError(f'Expected exactly one input for {name}; found {len(files)}.')
        path = files[0]
        row = load_npz(path)
        value = row['samples']
        # Published rows use the bundle digest. Fresh rows carry the current
        # scientific settings under a distinct key.
        if (checksum(path) != expected.get(path.name)
                and str(row.get('settings_sha256')) != settings_hash):
            raise ValueError(f'Unknown heatmap row data or settings: {path}')
        if (value.shape != (len(minimum['detuning']), len(minimum['k']), n)
                or not np.isfinite(value).all() or float(row['beta']) != beta
                or int(row['seed_start']) != int(minimum['seed'])+index*jobs):
            raise ValueError(f'Heatmap row settings or samples differ: {path}')
        sigma.append(value.mean(axis=-1))
        sem.append(value.std(axis=-1, ddof=1)/np.sqrt(n))
    minimum.update(sigma=np.asarray(sigma), sigma_sem=np.asarray(sem))
    save_npz(root/'data/minimum_grid.npz', minimum)
    print('Assembled mean EPR and SEM across the sampled coupling grid.', flush=True)


def theory_curves(root, deltas=None, ks=None):
    """Solve the canonical two-radius PDE at recorded main/inset settings."""
    params = load_parameters(HERE)
    fields = load_npz(HERE/'data/theory.npz')
    subset = deltas is not None or ks is not None
    selected = []
    for record in theory_records(params):
        kind, cfg = record['kind'], record['config']
        if ((deltas is not None and cfg['delta_omega'] not in deltas)
                or (ks is not None and cfg['k'] not in ks)):
            continue
        function = solve_pde if kind == 'main' else solve_pde_contributions
        values = function(**cfg)
        if subset:
            serial = {key: np.asarray(value).tolist() for key, value in values.items()}
            selected.append(dict(kind=kind, settings=cfg, result=serial))
        else:
            index = ((record['delta_index'], record['k_index']) if kind == 'main'
                     else record['k_index'])
            for name in fields:
                if name.startswith(kind+'_'):
                    fields[name][index] = values[name[len(kind)+1:]]
    if subset:
        if not selected:
            raise ValueError('No recorded PDE cases match --delta and --k.')
        write_json(root/'data/theory_cases.json', selected)
        print(f'Computed {len(selected)} selected stationary PDE cases.', flush=True)
    else:
        save_npz(root/'data/theory.npz', fields)


def weak_response_tables(root):
    """Recompute the two bimodal population slices in panel (c)."""
    b = load_npz(HERE/'data/bimodal.npz')
    settings = json.loads(str(b['settings']))
    nr = settings['nr']
    def weak(beta, detunings):
        T = 1/float(beta)
        return weak_pair_response(T, detunings, nr,
                radial_extent(T, settings['rmax']))['C2']
    c2_delta = weak(settings['beta_fixed'], b['delta'])
    c2_beta = np.array([weak(beta, [0., settings['delta_fixed']]) for beta in b['beta']])
    b.update(C2_0_fixed_beta=c2_delta[0], C2_delta_fixed_beta=c2_delta,
             C2_0_beta=c2_beta[:,0], C2_fixed_delta_beta=c2_beta[:,1],
             S_N_delta=np.column_stack([bimodal_coefficient(b['N'], c2_delta[0], c2) for c2 in c2_delta]),
             S_N_beta=np.column_stack([bimodal_coefficient(b['N'], *c2) for c2 in c2_beta]))
    save_npz(root/'data/bimodal.npz', b)


def compute(root, selected=None, inputs=()):
    """Run Heun trajectories from the recorded settings."""
    params = load_parameters(HERE)
    plans = simulation_records(params) + validation_runs(params)
    if selected is not None:
        chosen = [run for run in plans if run['id'] in selected]
        if {run['id'] for run in chosen} != set(selected):
            raise ValueError('Select recorded case IDs from data/parameters.json.')
        for run in chosen:
            result = simulate(**run['config'])
            save_npz(root/f"data/{run['id']}.npz", result)
        write_json(root/'data/cases.json', dict(
            cases=[{key: value for key, value in run.items()
                    if key in ('id', 'config', 'kind', 'base_id')} for run in chosen]))
        return
    replicas, variants = {}, {}
    for run in plans:
        result = simulate(**run['config'])
        name = run['id']
        destination = variants if 'base_id' in run else replicas
        destination.update({name+'_'+key: value for key, value in result.items()})
        print(f'Computed {name}', flush=True)
    data = root/'data'
    data.mkdir(parents=True)
    for name, values in (('replicas.npz', replicas), ('validation_replicas.npz', variants)):
        save_npz(data/name, values)
    for name in ('theory.npz','minimum_grid.npz','bimodal.npz','overlay.npz','numerics.csv'):
        choices = [folder/'data'/name for folder in inputs if (folder/'data'/name).exists()]
        if len(choices) > 1:
            raise ValueError(f'Multiple calculation inputs for {name}.')
        source = choices[0] if choices else HERE/'data'/name
        if not choices and checksum(source) != params['checksums'][name]:
            raise ValueError(f'Archive checksum mismatch: {name}')
        shutil.copyfile(source, data/name)
    params['checksums'] = {name: checksum(data/name) for name in params['checksums']}
    write_json(data/'parameters.json', params)
    save_npz(data/'figure.npz', figure_arrays(params, replicas, load_npz(data/'theory.npz')))


def replica_count_checks(baselines, variants, replicas, uncoupled):
    """Estimate uncertainty on nested subsets at each refinement point."""
    checked = {run['base_id'] for run in variants}
    rows = []
    for run in baselines:
        run_id = run['id']
        if run_id not in checked:
            continue
        delta, k = run['config']['delta_omega'], run['config']['k']
        values = replicas[run_id+'_values']
        for count in (16, 32):
            row = dict(run_id=run_id, delta_omega=delta, k=k, replicas=count)
            for name, column in ESTIMATOR_COLUMNS:
                row.update(estimate_fields(name, interval(values[:count, column]/uncoupled[delta])))
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plot','validate','compute','theory','weak-response',
                                           'overlay','heatmap','assemble-heatmap','numerics'))
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--case', nargs='+', help='Compute just the selected trajectory case IDs')
    parser.add_argument('--delta', type=float, nargs='+', help='Select saved PDE detunings')
    parser.add_argument('--k', type=float, nargs='+', help='Select saved PDE couplings')
    parser.add_argument('--beta-rows', type=int, nargs='+', help='Zero-based heatmap rows; 0–79')
    parser.add_argument('--inputs', type=Path, nargs='+', default=[],
                        help='Directories with calculated deterministic inputs or heatmap rows')
    parser.add_argument('--report', type=Path, help='Write recomputed validation report here (optional)')
    args = parser.parse_args(argv)
    if args.threads is not None:
        if args.threads < 1:
            parser.error('--threads must be positive.')
        set_num_threads(args.threads)
    if args.command == 'plot':
        plot(args.outdir or HERE)
    elif args.command == 'validate':
        validate(args.outdir or HERE, args.report)
    else:
        if args.outdir is None:
            parser.error('Calculations require a new --outdir.')
        root = fresh_output(args.outdir)
        if args.command == 'compute':
            compute(root, args.case, args.inputs)
        elif args.command == 'heatmap':
            rows = sorted(set(args.beta_rows)) if args.beta_rows is not None else None
            if rows is not None and not all(0 <= row < 80 for row in rows):
                parser.error('--beta-rows must select zero-based rows 0–79.')
            heatmap(root, rows)
        elif args.command == 'assemble-heatmap':
            if not args.inputs:
                parser.error('assemble-heatmap requires --inputs with complete raw rows.')
            assemble_heatmap(root, args.inputs)
        elif args.command == 'theory':
            theory_curves(root, args.delta, args.k)
        elif args.command == 'numerics':
            numerical_checks(root)
        elif args.command == 'weak-response':
            weak_response_tables(root)
        else:
            calculate_boundary(root)


from scipy.integrate import quad
from scipy.optimize import brentq


def numerical_checks(root):
    """Recalculate comparisons for the PDE curves, sampled map and weak response."""
    params = load_parameters(HERE)
    theory, figure, minimum = [load_npz(HERE/'data'/name) for name in
        ('theory.npz', 'figure.npz', 'minimum_grid.npz')]
    T, bar = params['model']['T_eff'], params['model']['omega_bar']
    deltas, ks = np.asarray(params['model']['deltas']), np.asarray(params['theory_k'])
    rows = []
    def record(check, quantity, baseline, refined, **settings):
        rows.append(dict(check=check, quantity=quantity, baseline=float(baseline),
                         refined=float(refined), absolute_difference=float(abs(refined-baseline)),
                         **settings))
    def pde(delta, k, nr=40, rmax=5., modes=48, temperature=T, **kwargs):
        return solve_pde(k=float(k), T_eff=float(temperature), omega_bar=bar,
                         delta_omega=float(delta), nr=nr, rmax=rmax, modes=modes, **kwargs)
    for delta, k in ((4.,1.), (4.,3.5), (4.,10.), (1.,1.), (1.,10.)):
        i, j = int(np.flatnonzero(deltas==delta)[0]), int(np.flatnonzero(ks==k)[0])
        base = float(theory['main_epr'][i,j])
        fine = float(pde(delta,k,nr=80)['epr'])
        extended = float(pde(delta,k,nr=88,rmax=6.)['epr'])
        record('pde_radial','epr',base,fine,beta=1/T,delta_omega=delta,k=k)
        record('pde_domain','epr',fine,extended,beta=1/T,delta_omega=delta,k=k)
        print(f'PDE refinement: Delta={delta:g}, k={k:g}',flush=True)
    for delta, k in ((4.,1.), (4.,3.5), (4.,10.), (1.,10.)):
        i, j = int(np.flatnonzero(deltas==delta)[0]), int(np.flatnonzero(ks==k)[0])
        refined = float(theory['inset_epr'][j]) if delta==4 else float(pde(delta,k,modes=96)['epr'])
        record('fourier','epr',theory['main_epr'][i,j],refined,beta=1/T,delta_omega=delta,k=k)
    for k in (3.5,10.):
        solution = pde(4.,k,modes=96,return_solution=True)
        parts = [decompose_pde_currents(solution,k,T,bar,4.,count)['current_terms'] for count in (512,1024)]
        j = int(np.flatnonzero(ks==k)[0])
        mean = float(figure['inset_theory_mean'][j])
        for quantity, values in (
                ('angular_variation_epr',[v[1:3].sum()-mean for v in parts]),
                ('radial_variation_epr',[v[3:5].sum() for v in parts])):
            record('phase_quadrature',quantity,*values,beta=1/T,delta_omega=4.,k=k)
    for beta, ds in ((.1,[0.,2.]), (1.,[0.,1.,2.,4.]), (2.,[0.,2.])):
        responses = [weak_pair_response(1/beta,ds,nr=nr,rmax=radial_extent(1/beta))['C2'] for nr in (80,120)]
        for i, delta in enumerate(ds):
            if beta!=1 and delta==0: continue
            for number in (2,30):
                values = [float(bimodal_coefficient(number,v[0],v[i])) for v in responses]
                record('weak_radial','S_N',*values,beta=beta,delta_omega=delta,number=number)
    for target in (1.,5.):
        bi = int(np.argmin(abs(minimum['beta']-target)))
        di = int(np.argmin(abs(minimum['detuning']-2.)))
        beta, delta = float(minimum['beta'][bi]), float(minimum['detuning'][di])
        grid, extent = np.asarray(minimum['k']), radial_extent(1/beta)
        values = np.array([float(pde(delta,k,temperature=1/beta,rmax=extent)['epr']) for k in grid])
        zero = int(np.flatnonzero(grid==0)[0]); index = int(np.argmin(values)); k = float(grid[index])
        sampled = minimum['sigma'][bi,di]; source = int(np.argmin(sampled))
        ratio = values[index]/values[zero]
        record('minimum_grid','normalized_epr',sampled[source]/sampled[zero],ratio,
               beta=beta,delta_omega=delta,k=k,sampled_k=float(grid[source]))
        fine = pde(delta,k,temperature=1/beta,nr=60,rmax=extent)['epr']
        fine_zero = pde(delta,0.,temperature=1/beta,nr=60,rmax=extent)['epr']
        record('minimum_radial','normalized_epr',ratio,fine/fine_zero,beta=beta,delta_omega=delta,k=k)
        print(f'Sampled minimum: beta={beta:g}, Delta={delta:g}',flush=True)
    boundary = params['overlay']['settings']
    def entry(nr):
        return brentq(lambda beta: float(weak_pair_response(1/beta,[boundary['upper']],nr,
            radial_extent(1/beta,tail_log=boundary['tail_log']))['C2'][0]),
            boundary['beta_min'],boundary['beta_max'],xtol=boundary['xtol'])
    record('boundary_entry','beta',entry(160),entry(200),delta_omega=boundary['upper'])
    for beta in (.4,1.045,10.):
        extent = radial_extent(1/beta,tail_log=boundary['tail_log'])
        def zero(nr,rmax):
            return weak_zero_detuning(1/beta,nr,rmax,boundary['upper'],boundary['xtol'])['detuning']
        base = zero(160,extent)
        record('boundary_radial','detuning',base,zero(200,extent),beta=beta)
        nr = int(np.ceil(160*np.sqrt(radial_extent(1/beta,tail_log=60)/extent)))
        extended = extent*(nr/160)**2
        record('boundary_domain','detuning',base,zero(nr,extended),beta=beta)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = root/'data/numerics.csv'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f'Saved {len(rows)} numerical comparisons in {path}',flush=True)
    return rows


"""Saved-data plotting for the phase-coupling figure; no numerical solvers."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# Previous orange/teal palette: ('#287C8E', '#C36D4E').
COLORS = ('#377EB8', '#DD5548', '#a52e3c')
INSET_MEAN_COLOR = '#4D64A8'
TICK_LABEL_SIZE, SMALL_TICK_LABEL_SIZE, COLORBAR_TICK_LABEL_SIZE = 15, 12, 13
AXIS_LABEL_SIZE, INSET_AXIS_LABEL_SIZE = 18, 14
LEGEND_SIZE, INSET_LEGEND_SIZE = 16, 16
INSET_BOUNDS = (.12, .55, .52, .43)
INSET_LINEWIDTH, INSET_MARKERSIZE, MAIN_MARKERSIZE = 1.6, 3.4, 5.5


def style_axes(ax, inset=False):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(.8)
    ax.tick_params(colors='black', direction='out', which='both',
                   labelsize=SMALL_TICK_LABEL_SIZE if inset else TICK_LABEL_SIZE)
    size = INSET_AXIS_LABEL_SIZE if inset else AXIS_LABEL_SIZE
    ax.xaxis.label.set_size(size)
    ax.yaxis.label.set_size(size)
    ax.grid(False, which='both', axis='both')


def label_zero_contour(ax, contour, label, fontsize, target=(.5, .5)):
    segments = [segment for segment in contour.allsegs[0] if len(segment) > 1]
    if not segments:
        return []
    points = np.concatenate(segments)
    lower = np.array([ax.get_xlim()[0], ax.get_ylim()[0]])
    span = np.array([np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())])
    relative = (points-lower)/span
    visible = np.all((relative >= 0) & (relative <= 1), axis=1)
    if not np.any(visible):
        return []
    point = points[visible][np.argmin(np.sum((relative[visible]-target)**2, axis=1))]
    labels = ax.clabel(contour, fmt={0: label}, inline=True, inline_spacing=6,
                       fontsize=fontsize, manual=[tuple(point)])
    for text in labels:
        text.set_alpha(1)
    return labels


def plot_heatmap(fig, ax, colorbar_ax, data, overlay):
    zero = np.flatnonzero(np.asarray(data['k']) == 0)
    if len(zero) != 1:
        raise ValueError('Minimum-EPR data must have exactly one k=0 entry.')
    sigma = data['sigma']
    ratio = sigma.min(axis=-1)/sigma[:, :, zero[0]]
    base = np.array(to_rgb(COLORS[0]))
    cmap = LinearSegmentedColormap.from_list('normalised_epr',
           [(0, .55*base), (.5, base), (.975, .2*base+.8), (1, COLORS[1])], N=1024)
    mesh = ax.pcolormesh(data['detuning'], data['beta'], ratio, shading='nearest',
                         cmap=cmap, norm=Normalize(vmin=ratio.min(), vmax=1))
    # Draw the directly solved boundary; the historical C2 grid is not used.
    from matplotlib.contour import ContourSet
    segment=np.column_stack((overlay['detuning'],overlay['beta']))
    contour=ContourSet(ax,[0.],[[segment]],colors=['white'],linestyles='--',linewidths=3.5)
    ax.set(xlabel=r'$\Delta\omega\ (\tau_r^{-1})$', ylabel=r'$\beta$',
           xlim=(data['detuning'].min(), data['detuning'].max()),
           ylim=(data['beta'].min(), data['beta'].max()))
    label_zero_contour(ax, contour, r'$C_2=0$', fontsize=18, target=(.4, .5))
    ax.text(.01, .975, r'$\mathrm{(b)}$', transform=ax.transAxes,
            fontsize=18, fontweight='normal', color='black', ha='left', va='top')
    style_axes(ax)
    colorbar = fig.colorbar(mesh, cax=colorbar_ax)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label(r'$\min_k\sigma(k)\ (\sigma(0))$', fontsize=AXIS_LABEL_SIZE)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    colorbar.ax.tick_params(colors='black', which='both', labelsize=COLORBAR_TICK_LABEL_SIZE)


def draw_bimodal_panels(fig, axes, data, settings, colorbar_ax):
    cmap = LinearSegmentedColormap.from_list('signed_epr', [COLORS[0], '#FBFAF7', COLORS[1]])
    limit = max(float(np.max(np.abs(data[name]))) for name in ('S_N_delta', 'S_N_beta'))
    limit = max(limit, np.finfo(float).eps)
    norm = (SymLogNorm(linthresh=settings['linthresh'], vmin=-limit, vmax=limit, base=10)
            if settings['color_scale'] == 'symlog'
            else TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit))
    number_top = min(18., float(data['N'].max()))
    for ax, x, values, fixed in [
            (axes[0], data['delta'], data['S_N_delta'], rf'$\beta={settings["beta_fixed"]:g}$'),
            (axes[1], data['beta'], data['S_N_beta'], rf'$\Delta={settings["delta_fixed"]:g}$')]:
        mesh = ax.pcolormesh(x, data['N'], values, shading='nearest', cmap=cmap, norm=norm)
        ax.set(xlim=(x.min(), x.max()), ylim=(data['N'].min(), number_top))
        if values.min() < 0 < values.max():
            zero = ax.contour(x, data['N'], values, levels=[0], colors='#202020',
                              linestyles='--', linewidths=2, alpha=1)
            label_zero_contour(ax, zero, r'$S_N=0$', fontsize=15)
        ax.set_title(fixed, fontsize=20, pad=8)
        ax.tick_params(colors='black', direction='out', labelsize=SMALL_TICK_LABEL_SIZE)
        for spine in ax.spines.values():
            spine.set_color('black')
        ax.grid(False)
    ticks = np.arange(data['N'].min(), number_top+1, 4)
    axes[0].set_yticks(np.unique(np.r_[data['N'][0], ticks, number_top]))
    axes[0].text(.015, 1.03, r'$\mathrm{(c)}$', transform=axes[0].transAxes,
                 fontsize=18, fontweight='normal', color='black', ha='left',
                 va='bottom', clip_on=False)
    for ax, label in zip(axes, (r'$\mathrm{(i)}$', r'$\mathrm{(ii)}$')):
        ax.text(.035, .965, label, transform=ax.transAxes, fontsize=16,
                fontweight='normal', color='black', ha='left', va='top')
    axes[0].set_xlabel(r'$\Delta\ (\tau_r^{-1})$', fontsize=AXIS_LABEL_SIZE)
    axes[0].set_ylabel(r'$N$', fontsize=AXIS_LABEL_SIZE)
    axes[1].set_xlabel(r'$\beta$', fontsize=AXIS_LABEL_SIZE)
    colorbar = fig.colorbar(mesh, cax=colorbar_ax)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label(r'$S_N$', fontsize=AXIS_LABEL_SIZE)
    colorbar.ax.tick_params(which='both', labelsize=COLORBAR_TICK_LABEL_SIZE)


def append_bimodal_panels(fig, data, settings):
    original_dpi = fig.dpi
    fig.set_dpi(300)
    fig.canvas.draw()
    width, original_height = fig.get_size_inches()
    positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
    heatmap_box, colorbar_box = positions[1][1], positions[2][1]
    gap, bottom_padding = .4, .68
    panel_width = (heatmap_box.width*width-gap)/2
    added_height = np.ceil(300*(panel_width+bottom_padding+.4))/300
    height = original_height+added_height
    fig.set_layout_engine('none')
    fig.set_size_inches(width, height)
    for ax, box in positions:
        ax.set_position([box.x0, (box.y0*original_height+added_height)/height,
                         box.width, box.height*original_height/height])
    left, bottom = heatmap_box.x0, bottom_padding/height
    panel_height = panel_width/height
    first = fig.add_axes([left, bottom, panel_width/width, panel_height])
    second = fig.add_axes([left+(panel_width+gap)/width, bottom,
                          panel_width/width, panel_height], sharey=first)
    second.tick_params(labelleft=False)
    colorbar_ax = fig.add_axes([colorbar_box.x0, bottom, colorbar_box.width, panel_height])
    draw_bimodal_panels(fig, [first, second], data, settings, colorbar_ax)
    fig.set_dpi(original_dpi)


def plot_decomposition(ax, data):
    deltas = np.asarray(data['deltas'])
    match = np.flatnonzero(deltas == 4.)
    if len(match) != 1:
        raise ValueError('Decomposition data require detuning 4 in the main panel.')
    baseline = data['baseline'][match[0]]
    psi, trajectory_psi = data['inset_theory_psi'], data['inset_sim_psi']
    handles = []
    for values, measured, errors, style, color, marker, label in [
            (data['inset_theory_epr'], data['inset_sim_epr'], data['inset_sim_epr_sem'],
             '-', COLORS[match[0]], 'o', r'$\sigma$'),
            (psi+data['inset_theory_mean'], trajectory_psi+data['inset_sim_mean'],
             data['inset_sim_mean_sem'], '--', INSET_MEAN_COLOR, 's',
             r'$\sigma_\psi+\sigma_{\Delta\Omega}$')]:
        ax.plot(data['theory_k'], values/baseline, style, color=color,
                linewidth=INSET_LINEWIDTH)
        ax.errorbar(data['main_k'], measured/baseline, yerr=errors/baseline,
                    fmt=marker, color=color, markerfacecolor='white',
                    markersize=INSET_MARKERSIZE, markeredgewidth=.9, capsize=1.4,
                    elinewidth=.85, zorder=4)
        handles.append(Line2D([], [], color=color, linestyle=style, marker=marker,
                              markerfacecolor='white', markeredgewidth=.9,
                              markersize=INSET_MARKERSIZE, linewidth=INSET_LINEWIDTH,
                              label=label))
    for values, style, color, label in [
            (data['inset_theory_angular'], '-', '#54876B', r'$\sigma_\psi+\sigma_{\mathrm{var}}$'),
            (data['inset_theory_radial'], ':', '#72678E', r'$\sigma_\psi+\sum_i\sigma_{r_i\varphi}$')]:
        line, = ax.plot(data['theory_k'], (psi+values)/baseline, style, color=color,
                        linewidth=INSET_LINEWIDTH, label=label)
        handles.append(line)
    theory_k = data['theory_k']
    ax.set(xlim=(theory_k.min(), theory_k.max()), xlabel=r'$k\ (\tau_r^{-1})$',
           ylabel=r'$\sigma\ (\sigma(0))$')
    ax.margins(y=.17)
    lower, upper = ax.get_ylim()
    ax.set_ylim(lower, upper+.55*(upper-lower))
    ax.set_xticks(np.linspace(theory_k.min(), theory_k.max(), 3))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.text(.97, .97, r'$\Delta\omega=4$', fontsize=14, transform=ax.transAxes,
            ha='right', va='top')
    style_axes(ax, inset=True)
    ax.legend(handles=handles, fontsize=INSET_LEGEND_SIZE, loc='upper center',
              bbox_to_anchor=(.43, 1), frameon=False, handlelength=2,
              handletextpad=.5, labelspacing=0)


def plot_main(figure, minimum, bimodal_data, settings, outdir, overlay):
    """Render the complete figure from saved arrays and save PNG/SVG only."""
    plt.rcParams['text.usetex'] = True
    plt.rcParams['legend.framealpha'] = 1
    for axis in ('xtick', 'ytick'):
        plt.rcParams[f'{axis}.major.size'] = 6
        plt.rcParams[f'{axis}.major.width'] = 1.1
        plt.rcParams[f'{axis}.minor.size'] = 4
    fig = plt.figure(figsize=(7.2, 10.8), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=(1, .045), hspace=.06, wspace=.05)
    ax = fig.add_subplot(grid[0, 0])
    heatmap_ax = fig.add_subplot(grid[1, 0])
    colorbar_ax = fig.add_subplot(grid[1, 1])
    handles = [Line2D([], [], color='black', linewidth=2, label=r'$\mathrm{theory}$')]
    has_zero = np.any(np.asarray(figure['deltas']) == 0)
    for j, (delta, color) in enumerate(zip(figure['deltas'], COLORS)):
        baseline = figure['baseline'][j]
        ratio = figure['pde_epr'][j]/baseline
        ax.plot(figure['theory_k'], ratio, color=color, linewidth=2)
        label_k = 7.4
        above = delta == 1
        offset = 18 if above else -10
        if has_zero and delta == 1:
            label_k, offset, above = 8.7, -20, False
        elif delta == 0:
            label_k, offset, above = 8.4, 17, True
        ax.annotate(rf'$\Delta\omega={delta:g}$',
                    (label_k, np.interp(label_k, figure['theory_k'], ratio)),
                    xytext=(0, offset), textcoords='offset points',
                    fontsize=16, color='black', ha='center',
                    va='bottom' if above else 'top')
        ax.errorbar(figure['main_k'], figure['main_mean'][j, :, 0]/baseline,
                    yerr=figure['main_sem'][j, :, 0]/baseline, fmt='o', color=color,
                    markerfacecolor='white', markeredgewidth=1, markersize=MAIN_MARKERSIZE,
                    capsize=2, elinewidth=.8, clip_on=False, label='Stratonovich')
        if delta != 0:
            bound = 2*float(figure['omega_bar'])**2*float(figure['m2'])/(float(figure['T_eff'])*baseline)
            bound_color = .8*np.array(to_rgb(color))+.2
            ax.axhline(bound, color=bound_color, linestyle='--', linewidth=1.1, alpha=1)
    handles.append(Line2D([], [], color='#334155', marker='o', markerfacecolor='white',
                          linestyle='none', markersize=MAIN_MARKERSIZE,
                          label=r'$\mathrm{Stratonovich}$'))
    if np.any(np.asarray(figure['deltas']) != 0):
        handles.append(Line2D([], [], color='#64748B', linestyle='--',
                              label=r'$\mathrm{EPR\ lower\ bound}$'))
    ax.axhline(1, color='#9CA9B5', linestyle=':', linewidth=.8, zorder=0)
    ax.set(xscale='linear', xlim=(figure['theory_k'].min(), figure['theory_k'].max()),
           xlabel=r'$k\ (\tau_r^{-1})$', ylabel=r'$\sigma\ (\sigma(0))$')
    ax.text(.99, .98, rf'$\beta={1/float(figure["T_eff"]):g}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=18)
    ax.text(.005, .975, r'$\mathrm{(a)}$', transform=ax.transAxes,
            fontsize=18, fontweight='normal', color='black', ha='left', va='top')
    style_axes(ax)
    ax.legend(handles=handles, loc='center',
              bbox_to_anchor=(.77, .26) if has_zero else (.76, .30),
              fontsize=LEGEND_SIZE, frameon=False, handlelength=1.9 if has_zero else 2.4)
    _, upper = ax.get_ylim()
    if has_zero:
        upper = max(np.max(figure['pde_epr']/figure['baseline'][:, None]),
                    np.max((figure['main_mean'][:, :, 0]+figure['main_sem'][:, :, 0])
                           /figure['baseline'][:, None]))
    ax.set_ylim(0, max(1, upper)*1.4)
    inset = ax.inset_axes(INSET_BOUNDS)
    plot_decomposition(inset, figure)
    plot_heatmap(fig, heatmap_ax, colorbar_ax, minimum, overlay)
    append_bimodal_panels(fig, bimodal_data, settings)
    for mesh in fig.findobj(QuadMesh):
        mesh.set_edgecolor('face')
        mesh.set_linewidth(.8)
        mesh.set_antialiased(False)
        mesh.set_snap(True)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir/'figure4.png', dpi=300, transparent=False, facecolor='white')
    fig.savefig(outdir/'figure4.svg', transparent=False, facecolor='white')
    return fig


if __name__ == "__main__":
    main()
