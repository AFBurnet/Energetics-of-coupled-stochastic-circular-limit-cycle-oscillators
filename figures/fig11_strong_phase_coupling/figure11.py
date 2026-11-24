#!/usr/bin/env python3
"""Figure 11: strong-coupling calculations, plotting and validation."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from numba import set_num_threads
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fig04_phase_coupling import numerics

SETTINGS = 'data/settings.json'
REFERENCE = 'data/reference.npz'
REFINEMENTS = 'validation/refinements.npz'
COEFFICIENTS = 'validation/coefficients.json'
INNER_CONSTANT = 'validation/inner_constant.json'
REPLICA_FIELDS = ('values', 'seeds', 'dt', 'actual_duration')
PDE_FIELDS = ('epr', 'n_modes')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def save_npz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def key(k):
    return f'k{int(k)}'


def unpack(archive, prefix, fields):
    return {field: archive[f'{prefix}_{field}'] for field in fields}


def store(archive, prefix, result):
    archive.update({f'{prefix}_{field}': value for field, value in result.items()})


def model_config(settings, k):
    return dict(k=k, T_eff=settings['T_eff'], omega_bar=settings['omega_bar'],
                delta_omega=settings['delta_omega'])


def run_configs(settings, group):
    variants = settings['trajectory_refinements' if group == 'simulation' else 'pde_refinements']
    for variant, overrides in variants.items():
        for k in settings['k']:
            config = model_config(settings, k) | settings[group] | overrides
            if group == 'pde':
                mode_set = overrides.get('mode_set', 'baseline')
                config.pop('mode_set', None)
                config['modes'] = settings['fourier_modes'][key(k)][mode_set]
            yield k, variant, config


def read_replicas(archive, prefix, config):
    record = unpack(archive, prefix, REPLICA_FIELDS)
    k = config['k']
    tag = int.from_bytes(hashlib.sha256(float(k).hex().encode()).digest()[:8], 'little')
    seeds = np.random.SeedSequence([config['seed'], tag]).generate_state(config['n_replicas'])
    wmax = max(abs(config['omega_bar'] - config['delta_omega'] / 2),
               abs(config['omega_bar'] + config['delta_omega'] / 2))
    dt = min(config['dt_max'], config['step_scale'] / (1 + 2 * k + wmax))
    values = record['values']
    if (values.shape != (len(seeds), len(numerics.MEASURES)) or not np.isfinite(values).all()
            or not np.array_equal(record['seeds'], seeds) or float(record['dt']) != dt
            or float(record['actual_duration']) != np.ceil(config['duration'] / dt) * dt):
        raise ValueError(f'Invalid saved replicas for {prefix}.')
    return values, record['seeds']


def interval(values):
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite independent replica estimates.')
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(t.ppf(.975, len(values) - 1) * sem)
    return dict(mean=mean, sem=sem, ci95_low=mean-half, ci95_high=mean+half)


def estimate(k, case, values):
    return dict(k=k, case=case, replicas=len(values),
                stratonovich_epr=interval(values[:, 0]),
                observable_epr=interval(values[:, 1]),
                stratonovich_minus_observable=interval(values[:, 0] - values[:, 1]))


def paired_difference(k, case, values, baseline):
    return dict(k=k, case=case, replicas=len(values),
                stratonovich_epr_change=interval(values[:, 0] - baseline[:, 0]))


def pde_checks(settings, reference, refinements):
    checks = []

    def append(k, case, config, result, baseline):
        epr = float(result['epr'])
        checks.append(dict(k=k, case=case, stationary_pde_epr=epr,
                           relative_change=(epr - baseline) / baseline,
                           nr=config['nr'], modes=int(result['n_modes']), rmax=config['rmax']))

    for k in settings['k']:
        baseline = unpack(reference, f'{key(k)}_pde', PDE_FIELDS)
        append(k, 'baseline', settings['pde'], baseline, float(baseline['epr']))
    for k, case, config in run_configs(settings, 'pde'):
        result = unpack(refinements, f'{case}_{key(k)}', PDE_FIELDS)
        append(k, case, config, result, float(reference[f'{key(k)}_pde_epr']))
    return checks


def asymptotic_comparison(settings, checks):
    from plotting import asymptotes

    rows = []
    for row in checks:
        if row['case'] != 'baseline':
            continue
        k, epr = row['k'], row['stationary_pde_epr']
        leading, two_term = asymptotes(k, settings['T_eff'], settings['theory'])
        rows.append(dict(k=k, stationary_pde_epr=epr, leading=leading, two_term=two_term,
                         leading_relative_difference=(leading-epr)/epr,
                         two_term_relative_difference=(two_term-epr)/epr))
    return rows


def summarize(settings, reference, refinements):
    estimates, comparisons, baselines = [], [], {}
    for k in settings['k']:
        config = model_config(settings, k) | settings['simulation']
        values, seeds = read_replicas(reference, key(k), config)
        baselines[k] = values, seeds
        estimates.append(estimate(k, 'baseline', values))

    for k, case, config in run_configs(settings, 'simulation'):
        values, seeds = read_replicas(refinements, f'{case}_{key(k)}', config)
        baseline, baseline_seeds = baselines[k]
        if not np.array_equal(seeds, baseline_seeds):
            raise ValueError(f'Replica seeds do not match the baseline for {case} at k={k:g}.')
        comparisons.append(paired_difference(k, case, values, baseline))

    checks = pde_checks(settings, reference, refinements)
    return dict(
        normalization=dict(uncoupled_epr=settings['sigma0']),
        uncertainty=('EPR is in simulation units. The manuscript divides by uncoupled EPR and shows one SEM. '
                     'Checks report pointwise 95% Student intervals. The two estimators share paths. '
                     'Trajectory changes are refined minus baseline, paired by seed; changing the '
                     'timestep gives different Brownian paths.'),
        estimator_benchmarks=estimates,
        trajectory_refinements=comparisons,
        stationary_pde_checks=checks,
        asymptotic_comparison=asymptotic_comparison(settings, checks),
    )


def fresh_output(root):
    root = root.resolve()
    if root == HERE or HERE in root.parents or root.exists() and any(root.iterdir()):
        raise ValueError('Choose an empty output directory outside the bundled figure.')
    root.mkdir(parents=True, exist_ok=True)
    return root


def compute(root, settings, pde_only=False):
    from coefficient_checks import run_checks

    reference, refinements = {}, {}
    for k in settings['k']:
        config = model_config(settings, k)
        store(reference, f'{key(k)}_pde', numerics.solve_pde(**(config | settings['pde'])))
        if not pde_only:
            store(reference, key(k), numerics.simulate(**(config | settings['simulation'])))
    for k, case, config in run_configs(settings, 'simulation'):
        store(refinements, f'{case}_{key(k)}', numerics.simulate(**config))
        print(f'Computed {case} at k={k:g}.', flush=True)
    for k, case, config in run_configs(settings, 'pde'):
        store(refinements, f'{case}_{key(k)}', numerics.solve_pde(**config))
        print(f'Computed {case} at k={k:g}.', flush=True)

    write_json(root / SETTINGS, settings)
    save_npz(root / REFERENCE, reference)
    save_npz(root / REFINEMENTS, refinements)
    if pde_only:
        checks = pde_checks(settings, reference, refinements)
        write_json(root / 'validation/pde.json', dict(stationary_pde_checks=checks))
    else:
        evidence = read_json(HERE / INNER_CONSTANT)
        write_json(root / INNER_CONSTANT, evidence)
        write_json(root / COEFFICIENTS, run_checks(settings['theory'], settings['T_eff'],
                                                 settings['omega_bar'], evidence))
        write_json(root / 'validation/report.json', summarize(settings, reference, refinements))
    print(f'Computed {len(settings["k"])} baseline PDE solutions.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plot', 'validate', 'compute'))
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--k', type=float, nargs='+', help='Compute only these recorded couplings')
    parser.add_argument('--pde-only', action='store_true', help='Compute without trajectories')
    parser.add_argument('--baseline-only', action='store_true', help='Skip refinement jobs')
    parser.add_argument('--report', type=Path, help='Save the recomputed numerical validation report')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if args.command == 'compute':
        if args.outdir is None:
            parser.error('Compute requires a new --outdir.')
        settings = read_json(HERE / SETTINGS)
        chosen = args.k or settings['k']
        if not set(chosen).issubset(settings['k']) or len(set(chosen)) != len(chosen):
            parser.error('--k must select distinct recorded couplings.')
        settings['k'] = chosen
        if args.baseline_only or args.pde_only:
            settings['trajectory_refinements'] = {}
        if args.baseline_only:
            settings['pde_refinements'] = {}
        set_num_threads(args.threads)
        compute(fresh_output(args.outdir), settings, args.pde_only)
        return

    root = (args.outdir or HERE).resolve()
    settings, reference = read_json(root / SETTINGS), read_npz(root / REFERENCE)
    if any(f'{key(k)}_values' not in reference for k in settings['k']):
        parser.error('A PDE-only calculation reports results in validation/pde.json.')
    if args.command == 'plot':
        from plotting import plot_main
        plot_main(root, settings, reference)
        return

    from coefficient_checks import check_inner_constant
    check_inner_constant(settings['theory']['inner_C'], read_json(root / INNER_CONSTANT))
    report = summarize(settings, reference, read_npz(root / REFINEMENTS))
    print(f"Replayed {len(report['estimator_benchmarks'])} EPR benchmarks, "
          f"{len(report['trajectory_refinements'])} trajectory refinements and "
          f"{len(report['stationary_pde_checks'])} PDE records.")
    if args.report:
        write_json(args.report, report)


if __name__ == '__main__':
    main()
