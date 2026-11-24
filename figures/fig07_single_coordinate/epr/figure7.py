#!/usr/bin/env python3
"""Figure 7: single-coordinate Cartesian coupling, saved simulations and checks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'energetics-figure7-matplotlib'))
import numpy as np
from numba import set_num_threads
from scipy.stats import t as student_t
import single_cartesian_numerics as numerics

HERE = Path(__file__).resolve().parent
LONG_RUN_BETAS = (*range(50, 101, 5), .1, .5, 1., 10/3, 7., 10., 16., 25., 40.)
RAW_FIELDS = ('work', 'observable', 'replica_seeds', 'block_values', 'block_time_edges')
ESTIMATORS = {'work': 'stratonovich_epr', 'observable': 'observable_epr'}
REPORT_NORMALIZATION = 'EPR estimates and differences are divided by the total uncoupled EPR, N sigma_0.'
REPORT_UNCERTAINTY = ('Pointwise 95% Student intervals across independent complete oscillator systems. '
    'The two EPR estimators share trajectories. Changes between runs use independent-sample Welch intervals.')
REPORT_BLOCKS = ('Late minus early EPR is paired across measurement halves. '
    'The halves are duration-weighted within each replica.')
REPORT_REFERENCE = 'At k=0 the stationary normalized EPR is one.'


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_npz(path, values):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def load_parameters(root):
    return json.loads((root/'data/parameters.json').read_text())


def run_configs(plan):
    return {run['id']: {**plan['run_defaults'], **{key: value for key, value in run.items() if key != 'id'}}
            for run in plan['runs']}


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Need two finite independent systems')
    sem = float(values.std(ddof=1)/math.sqrt(len(values)))
    return dict(mean=float(values.mean()), sem=sem, half95=float(student_t.ppf(.975, len(values)-1)*sem))


def independent_difference(refined, baseline):
    """Refined minus baseline with an independent-sample Welch interval."""
    a, b = np.asarray(refined), np.asarray(baseline)
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    sem = math.sqrt(va+vb)
    df = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)) if sem else 1e30
    return dict(mean=float(a.mean()-b.mean()), sem=sem, half95=float(student_t.ppf(.975, df)*sem))


def estimate_fields(name, estimate):
    return {name: estimate['mean'], name+'_sem': estimate['sem'],
            name+'_ci95_half_width': estimate['half95']}


def read_replicas(root, plan, configs):
    path = root/'data/replicas.npz'
    if not path.is_file() or checksum(path) != plan['checksums']['replicas.npz']:
        raise ValueError('Missing replica data or checksum mismatch: replicas.npz')
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for run_id, config in configs.items():
            n, nblocks = config['replicas'], config['blocks']
            values = {name: archive[f'{run_id}_{name}'] for name in RAW_FIELDS}
            # Earlier companion runs retained only the two EPR estimators.
            if f'{run_id}_second' in archive:
                values['second'] = archive[f'{run_id}_second']
            values['block_values'] = values['block_values'][..., :2]
            for name, entry in values.items():
                if name == 'block_values':
                    valid_shape = entry.shape == (n, nblocks, 2)
                else:
                    valid_shape = entry.shape == ((nblocks+1,) if name == 'block_time_edges' else (n,))
                if not valid_shape or not np.isfinite(entry).all():
                    raise ValueError(f'Invalid replica field: {run_id}_{name}')
            if len(np.unique(values['replica_seeds'])) != n:
                raise ValueError(f'Duplicate replica seeds in {run_id}')
            edges = values['block_time_edges']
            widths = np.diff(edges)
            if edges[0] != 0 or np.any(widths <= 0):
                raise ValueError(f'Invalid measurement block edges in {run_id}')
            for column, name in enumerate(ESTIMATORS):
                combined = np.average(values['block_values'][:, :, column], axis=1, weights=widths)
                if not np.allclose(combined, values[name], rtol=1e-8, atol=1e-9):
                    raise ValueError(f'Measurement blocks disagree with complete-system estimates in {run_id}')
            result[run_id] = values
    return result


def compute(root, threads=None, cases=None, *, plateau=False):
    """Calculate the recorded EPR or longer-run simulations."""
    plan = load_parameters(HERE/'long_runs' if plateau else HERE)
    configs = run_configs(plan)
    cases = set(cases or ())
    unknown = cases-configs.keys()
    if unknown:
        raise ValueError('Unknown run IDs: '+', '.join(sorted(unknown)))
    runs = {run_id: config for run_id, config in configs.items() if not cases or run_id in cases}
    if threads is not None:
        set_num_threads(threads)
    archive = {}
    for run_id, config in runs.items():
        values = numerics.simulate_replicas(**config)
        archive.update({f'{run_id}_{name}': values[name] for name in (*RAW_FIELDS, 'second')})
        print(f"Calculated {run_id}: N={config['number']}, beta={config['beta']:g}, k={config['k']:g}", flush=True)
    data = root/'data'
    data.mkdir(parents=True)
    save_npz(data/'replicas.npz', archive)
    plan['checksums'] = {'replicas.npz': checksum(data/'replicas.npz')}
    if runs.keys() != configs.keys():
        plan['completed_runs'] = list(runs)
    write_json(data/'parameters.json', plan)


def normalization(config):
    return config['number']*config['beta']*config['omega']**2*numerics.free_second(config['beta'])


def benchmark(run_id, config, values, norm):
    row = dict(run_id=run_id, number=config['number'], beta=config['beta'], k=config['k'],
               replicas=config['replicas'])
    for field, name in ESTIMATORS.items():
        row.update(estimate_fields(name, interval(values[field]/norm)))
    row.update(estimate_fields('stratonovich_minus_observable',
                               interval((values['work']-values['observable'])/norm)))
    widths = np.diff(values['block_time_edges'])
    half = config['blocks']//2
    blocks = values['block_values']
    late = np.average(blocks[:, half:, 0], axis=1, weights=widths[half:])
    early = np.average(blocks[:, :half, 0], axis=1, weights=widths[:half])
    row.update(estimate_fields('stratonovich_epr_late_minus_early', interval((late-early)/norm)))
    return row


def run_difference(archive, baseline, refined, norm):
    a, b = archive[baseline], archive[refined]
    if np.intersect1d(a['replica_seeds'], b['replica_seeds']).size:
        raise ValueError(f'Comparison streams overlap: {baseline}, {refined}')
    return estimate_fields('stratonovich_epr_change',
                           independent_difference(b['work']/norm, a['work']/norm))


def figure_summary(plan, configs, benchmarks, archive, norms):
    plateau_k = plan['plateau_k']
    plateau_ids = [run_id for run_id in plan['panels']['b'] if configs[run_id]['k'] == plateau_k]
    models = {(configs[run_id]['number'], configs[run_id]['omega']) for run_id in plateau_ids}
    if len(models) != 1:
        raise ValueError('Main EPR curve requires a common population size and intrinsic frequency')
    number, omega = models.pop()
    panels = {name: ids for name, ids in plan['panels'].items() if name != 'b' and ids}
    if 'coupling_checks' not in plan:
        panels['b'] = plateau_ids
    shown = {run_id for ids in panels.values() for run_id in ids}
    # Only the plateau and the beta=10/3 coupling sweep enter panel (c).
    decomposition_ids = set(panels.get('b', ())) | {
        run_id for run_id in panels.get('a', ())
        if configs[run_id]['number'] == 2 and np.isclose(configs[run_id]['beta'], 10/3)
    }
    rows = []
    for run_id, row in benchmarks.items():
        if run_id in shown:
            point = dict(id=run_id, number=row['number'], beta=row['beta'], k=row['k'],
                stratonovich_epr=row['stratonovich_epr'],
                stratonovich_epr_sem=row['stratonovich_epr_sem'])
            if run_id in decomposition_ids and 'second' in archive[run_id]:
                values, config, norm = archive[run_id], configs[run_id], norms[run_id]
                radial = config['beta']*config['omega']**2*config['number']*values['second']/norm
                # Subtract within each replica before computing the standard error.
                correction = radial - values['observable']/norm
                for name, samples in (('radial_reference', radial), ('correction', correction)):
                    estimate = interval(samples)
                    point[name], point[name+'_sem'] = estimate['mean'], estimate['sem']
                point.update(observable_epr=row['observable_epr'], observable_epr_sem=row['observable_epr_sem'])
            rows.append(point)
    return dict(panels=panels, plateau_k=plateau_k,
                main_curve_model=dict(number=number, omega=omega), rows=rows)


def validate(root):
    plan = load_parameters(root)
    if 'completed_runs' in plan:
        raise ValueError('Validation requires the complete simulation plan')
    configs = run_configs(plan)
    archive = read_replicas(root, plan, configs)
    norms = {run_id: normalization(config) for run_id, config in configs.items()}
    benchmarks = {run_id: benchmark(run_id, config, archive[run_id], norms[run_id])
                  for run_id, config in configs.items()}
    report = dict(normalization=REPORT_NORMALIZATION, uncertainty=REPORT_UNCERTAINTY,
                  measurement_halves=REPORT_BLOCKS,
                  exact_benchmark=REPORT_REFERENCE, estimator_benchmarks=list(benchmarks.values()))
    if 'refinements' in plan:
        refinements = []
        for check in plan['refinements']:
            baseline, refined = check['baseline_run'], check['refined_run']
            config = configs[baseline]
            row = dict(changed=check['changed'], baseline_run=baseline, refined_run=refined,
                       number=config['number'], beta=config['beta'], k=config['k'])
            row.update(run_difference(archive, baseline, refined, norms[baseline]))
            refinements.append(row)
        report['simulation_refinements'] = refinements
    if 'coupling_checks' in plan:
        coupling_rows = {(configs[run_id]['beta'], configs[run_id]['k']): run_id
                         for run_id in plan['panels']['b']}
        comparisons = []
        couplings = plan['coupling_checks']
        for beta in sorted({beta for beta, _ in coupling_rows}):
            for ka, kb in zip(couplings[:-1], couplings[1:]):
                baseline, refined = coupling_rows[beta, ka], coupling_rows[beta, kb]
                row = dict(beta=beta, k_from=ka, k_to=kb)
                row.update(run_difference(archive, baseline, refined, norms[baseline]))
                comparisons.append(row)
        report['finite_coupling_comparisons'] = comparisons
    write_json(root/'data/validation.json', report)
    write_json(root/'data/figure.json', figure_summary(plan, configs, benchmarks, archive, norms))
    print(f'Saved {len(benchmarks)} simulation summaries, '
          f'{len(report.get("simulation_refinements", []))} refinements, and '
          f'{len(report.get("finite_coupling_comparisons", []))} finite-coupling comparisons.', flush=True)


def load_figure(root, plateau_root):
    """Use the longer k=160 estimates for the main curve."""
    figure = json.loads((root/'data/figure.json').read_text())
    longer = json.loads((plateau_root/'data/figure.json').read_text())
    lookup = {row['id']: row for row in longer['rows']}
    replacements = [dict(lookup[run_id], id='plateau_'+run_id) for run_id in longer['panels']['b']]
    if sorted(row['beta'] for row in replacements) != sorted(LONG_RUN_BETAS):
        raise ValueError('Finish all planned long-run points before plotting')
    if (longer['plateau_k'] != figure['plateau_k']
            or longer['main_curve_model'] != figure['main_curve_model']
            or any(row['number'] != figure['main_curve_model']['number'] or row['k'] != figure['plateau_k']
                   for row in replacements)):
        raise ValueError('Long-run model parameters do not match the figure')
    figure['panels']['b'] = [row['id'] for row in replacements]
    figure['rows'].extend(replacements)
    return figure


def plot(root, plateau_root=None, dynamics_root=None):
    """Read saved EPR, trace and effective frequency summaries."""
    out = root/'figures'
    figure = load_figure(root, Path(plateau_root or HERE/'long_runs'))
    dynamics_root = Path(dynamics_root or HERE.parent/'frequency')
    frequency = json.loads((dynamics_root/'data/figure.json').read_text())
    from plot_figure7 import plot_figure
    plot_figure(out, figure, frequency)
    print(f'Saved PNG/SVG figures in {out}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', default='plot', choices=('plot', 'compute', 'validate', 'compute-plateau'))
    parser.add_argument('--outdir', type=Path, help='Data directory. Compute requires a separate, empty directory.')
    parser.add_argument('--threads', type=int, help='Numba threads during simulation')
    parser.add_argument('--case', action='append', help='Compute only this recorded run ID (repeatable)')
    parser.add_argument('--plateau-outdir', type=Path, help='Directory with the long-run EPR data for plot')
    parser.add_argument('--dynamics-outdir', type=Path, help='Directory with the effective frequency data for plot')
    args = parser.parse_args(argv)
    root = (args.outdir or HERE).resolve()
    compute_commands = ('compute', 'compute-plateau')
    if args.case and args.command not in compute_commands:
        parser.error('--case requires a computation command')
    if args.command in compute_commands:
        if args.outdir is None or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
            parser.error('Choose a separate, empty --outdir for new calculations')
        compute(root, args.threads, args.case, plateau=args.command=='compute-plateau')
    elif args.command == 'validate':
        validate(root)
    else:
        plot(root, args.plateau_outdir, args.dynamics_outdir)


if __name__ == '__main__':
    main()
