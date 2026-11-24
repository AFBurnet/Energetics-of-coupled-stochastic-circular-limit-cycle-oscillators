#!/usr/bin/env python3
"""Effective frequency analysis and numerical validation for Figure 7."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from numba import set_num_threads
from scipy.stats import t as student_t
import frequency_numerics as numerics

HERE = Path(__file__).resolve().parent


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run_configs(params):
    defaults = params['run_defaults']
    configs = {run['id']: {**defaults, **{key: value for key, value in run.items() if key != 'id'}}
               for run in params['runs']}
    return configs


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Require at least two finite independent-system estimates')
    sem = float(values.std(ddof=1)/math.sqrt(len(values)))
    return dict(mean=float(values.mean()), sem=sem,
                half95=float(student_t.ppf(.975, len(values)-1)*sem), count=len(values))


def difference(a, b):
    """Independent-sample Welch interval for mean(a)-mean(b)."""
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b)
    sem = math.sqrt(va+vb)
    df = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)) if sem else 1e30
    return dict(mean=float(a.mean()-b.mean()), sem=sem,
                half95=float(student_t.ppf(.975, df)*sem))


def compute(root, threads=None, cases=None):
    """Estimate effective frequency at the recorded manuscript settings and seeds."""
    root = Path(root)
    params = json.loads((HERE/'data/parameters.json').read_text())
    configs = run_configs(params)

    cases = set(cases or ())
    unknown = cases - configs.keys()
    if unknown:
        raise ValueError("Unknown run IDs: " + ", ".join(sorted(unknown)))
    runs = {identifier: config for identifier, config in configs.items()
            if not cases or identifier in cases}

    if threads is not None:
        set_num_threads(threads)
    raw = root/'data/runs'
    raw.mkdir(parents=True)
    checksums = {}
    for identifier, config in runs.items():
        result = numerics.run_frequency(config)
        archive = raw/f'{identifier}.npz'
        np.savez_compressed(archive, **result)
        checksums[identifier] = checksum(archive)
        print(f"Calculated {identifier}: N={config['number']}, beta={config['beta']:g}, k={config['k']:g}",
              flush=True)
    params['checksums'] = checksums
    if cases:
        params['completed_runs'] = list(runs)
        print("Selected runs only; this is not a complete figure data set.")
    write_json(root/'data/parameters.json', params)


def load_run(root, identifier, config, digest):
    path = root/'data/runs'/f'{identifier}.npz'
    if not path.is_file() or checksum(path) != digest:
        raise ValueError(f'Missing data or checksum mismatch: {path}')
    n, blocks = config['replicas'], config['blocks']
    shapes = dict(seeds=(n,), block_time_edges=(blocks+1,), block_frequency=(n, blocks),
                  diagnostics=(n, 2))
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in shapes}
        if result['diagnostics'].shape == (n, 5):
            result['diagnostics'] = result['diagnostics'][:, [1, 4]]
    for key, shape in shapes.items():
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError(f'Invalid numerical array: {identifier}/{key}')
    if len(np.unique(result['seeds'])) != n:
        raise ValueError(f'Duplicate replica seeds in {identifier}')
    measure_steps = round(config['measure']/config['dt'])
    edges = result['block_time_edges']
    if (edges[0] != 0 or np.any(np.diff(edges) <= 0)
            or not np.isclose(edges[-1], measure_steps*config['dt'], rtol=2e-13, atol=2e-13)):
        raise ValueError(f'Invalid measurement block edges in {identifier}')
    return result, measure_steps


def estimate_fields(name, estimate):
    return {name: estimate['mean'], name+'_sem': estimate['sem'],
            name+'_ci95_half_width': estimate['half95']}


def rates(data):
    return np.average(data['block_frequency'], axis=1, weights=np.diff(data['block_time_edges']))


def validate(root):
    root = Path(root)
    params = json.loads((root/'data/parameters.json').read_text())
    configs = run_configs(params)
    completed = set(params.get('completed_runs', configs))
    raw, rows = {}, []
    for identifier, config in configs.items():
        if identifier not in completed:
            continue
        data, measure_steps = load_run(root, identifier, config, params['checksums'][identifier])
        raw[identifier] = data
        values = rates(data)
        widths = np.diff(data['block_time_edges'])
        half = len(widths)//2
        late = np.average(data['block_frequency'][:, half:], axis=1, weights=widths[half:])
        early = np.average(data['block_frequency'][:, :half], axis=1, weights=widths[:half])
        row = dict(run_id=identifier, **{key: config[key] for key in
                   ('number', 'beta', 'k', 'omega', 'initial')})
        row.update(estimate_fields('frequency', interval(values)))
        row.update(estimate_fields('frequency_late_minus_early', interval(late-early)))
        steps = measure_steps*config['number']
        diagnostics = data['diagnostics']
        row['phase_resolution'] = dict(
            large_increment_fraction=interval(diagnostics[:, 0]/steps),
            exact_origin_endpoint_count=int(diagnostics[:, 1].sum()))
        rows.append(row)
    indexed = {row['run_id']: row for row in rows}
    comparisons = []
    for check in params['checks']:
        if check['base'] not in raw:
            continue
        if check['target'] in raw:
            base, target = raw[check['base']], raw[check['target']]
            if np.intersect1d(base['seeds'], target['seeds']).size:
                raise ValueError('Refinement comparisons require independent random streams')
            changed = {'dt': 'timestep', 'burn': 'burn_in', 'measure': 'measurement_duration',
                       'initial': 'initial_state', 'initial_long': 'initial_state_long_duration'}
            comparisons.append(dict(changed=changed[check['kind']],
                baseline_run=check['base'], refined_run=check['target'],
                **estimate_fields('frequency_change', difference(rates(target), rates(base)))))
    curves = []
    for curve in params['curves']:
        points = []
        for identifier in curve['points']:
            row = indexed.get(identifier)
            if row:
                points.append(dict(k=row['k'], initial=row['initial'],
                                   mean=row['frequency'], sem=row['frequency_sem']))
        curves.append(dict(curve, points=points))
    summary = dict(threshold=numerics.stability_threshold(), curves=curves)
    write_json(root/'data/figure.json', summary)
    result = dict(frequency_estimates=rows, simulation_refinements=comparisons,
                  uncertainty=params['uncertainty'],
                  normalization='Effective frequency estimates and differences are divided by omega.',
                  interpretation='Measurement halves are paired within each independent system. '
                  'Initial ensembles are reported separately.')
    write_json(root/'data/validation.json', result)
    print(f'Saved {len(rows)} runs and {len(comparisons)} convergence comparisons.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="validate", choices=("compute", "validate"))
    parser.add_argument("--outdir", type=Path, help="Data directory; compute requires a separate, empty directory")
    parser.add_argument("--threads", type=int, help="Numba threads during simulation")
    parser.add_argument("--case", action="append", help="Compute only this recorded run ID (repeatable)")

    args = parser.parse_args(argv)
    root = (args.outdir or HERE).resolve()
    if args.case and args.command != "compute":
        parser.error("--case requires a computation command")
    if args.command == "compute":
        if args.outdir is None or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
            parser.error("Choose a separate, empty --outdir for new calculations")
        compute(root, args.threads, args.case)
    elif args.command == "validate":
        validate(root)


if __name__ == '__main__':
    main()
