#!/usr/bin/env python3
"""Figure 3: N radially coupled oscillators (m=n=C=r_c=1).

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
from functools import lru_cache

import mpmath as mp
import numpy as np
from numba import njit, prange, set_num_threads
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import t as student_t

NUMBERS = (2, 4, 8, 16)
RAW_FIELDS = ('work', 'moment', 'replica_seeds')


def _log_I(beta, order):
    """log integral r**(order+1) exp[beta*(r**2/2-r**3/3)] dr.

    Positive 2F2 expression, evaluated inside the caller's mp.workdps context.
    """
    b, n = mp.mpf(beta), mp.mpf(order)
    if b <= 0 or n <= -2:
        raise ValueError('Require beta > 0 and moment order > -2')
    one_third = mp.mpf(1)/3
    z = b/6
    terms = (
        8*mp.gamma((n+2)/3)*mp.hyper([(n+2)/6, (n+5)/6],
                                    [one_third, 2*one_third], z),
        4*mp.power(3, 2*one_third)*mp.power(b, one_third)
        *mp.gamma((n+4)/3)*mp.hyper([(n+4)/6, (n+7)/6],
                                   [2*one_third, 4*one_third], z),
        3*mp.power(3, one_third)*mp.power(b, 2*one_third)
        *mp.gamma((n+6)/3)*mp.hyper([(n+6)/6, (n+9)/6],
                                   [4*one_third, 5*one_third], z),
    )
    return (-mp.log(8)+(n-1)*mp.log(3)/3-(n+2)*mp.log(b)/3
            +mp.log(mp.fsum(terms)))


@lru_cache(maxsize=1024)
def _log_free_second(beta, dps):
    with mp.workdps(dps):
        return _log_I(beta, 2)-_log_I(beta, 0)


def strong_ratio(beta, number, dps=35):
    """Calculate sigma_inf/(N*sigma0), extended to real N >= 1."""
    beta, number = float(beta), float(number)
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(number) or number < 1:
        raise ValueError('Require finite beta > 0 and N >= 1')
    with mp.workdps(dps):
        b, n = mp.mpf(beta), mp.mpf(number)
        log_ratio = (_log_I(n*b, n+1)-_log_I(n*b, n-1)
                     -_log_free_second(beta, dps))
        value = float(mp.exp(log_ratio))
    if not math.isfinite(value) or value <= 0:
        raise ArithmeticError('Invalid strong-coupling ratio')
    return value


def free_second(beta, dps=35):
    """Analytical uncoupled second radial moment."""
    with mp.workdps(dps):
        return float(mp.exp(_log_free_second(float(beta), dps)))


def unity_beta(number, bounds=(0.1, 10.0), dps=35):
    """Positive-temperature crossing sigma_inf/(N*sigma0)=1 for N>1."""
    if number <= 1:
        raise ValueError('The unity boundary requires N > 1')
    return brentq(lambda beta: strong_ratio(beta, number, dps)-1.0,
                  *bounds, xtol=2e-11, rtol=2e-13)


def strong_contour(betas, numbers, dps=35):
    """Grid indexed [beta, N], with free-system moments cached per beta."""
    return np.array([[strong_ratio(beta, number, dps) for number in numbers]
                     for beta in betas])


def _benchmark_free_second(beta, eps=1e-11, rmax=7.0):
    def term(r,p):
        return r**p*math.exp(beta*(r*r/2-r*r*r/3)-beta/6)
    return quad(term,0,rmax,args=(3,),epsabs=eps,epsrel=eps)[0]/quad(term,0,rmax,args=(1,),epsabs=eps,epsrel=eps)[0]


def stationary_ratio(beta, k, number, eps=1e-10, rmax=7.0, width=12.0):
    """sigma/(N*sigma0) by independently adaptive stationary quadrature.

    J_p(m)=int_0^infty r^(p+1) exp[-beta*u(r)-beta*k*N*(r-m)^2/2] dr.
    <r_i^2> = int_R J_0(m)^(N-1) J_2(m) dm / int_R J_0(m)^N dm.
    Here u(r)=r^3/3-r^2/2. Integrating the product of N Gaussians over
    m gives exp[-beta*k/2 sum_{i<j}(r_i-r_j)^2] times a constant;
    thus this is the finite-N stationary density, not a coupling limit.
    The two finite numerical domains are configurable and must be refined.
    Inner and outer exponential shifts avoid raising tiny J_0 values to N.
    Defaults are used at beta=0.8; the theory command saves refinement results.
    """
    N = number
    if beta<=0 or k<0 or N<2 or not math.isfinite(beta+k):
        raise ValueError('Need beta>0, k>=0, and N>=2')
    if k==0:
        return 1.0
    c=beta*k*N

    @lru_cache(maxsize=8192)
    def conditional(m):
        def score(r):
            return 1/r+beta*(r-r*r)-c*(r-m)
        mode=(brentq(score,1e-14,rmax,xtol=1e-14)
              if score(rmax)<0 else rmax)
        peak=math.log(mode)+beta*(mode*mode/2-mode**3/3)-c*(mode-m)**2/2
        curvature=1/(mode*mode)+beta*(2*mode-1)+c
        w=1/math.sqrt(max(curvature,1e-10))
        splits=sorted(set([0.,rmax]+[min(rmax,max(0.,mode+j*w)) for j in (-4.,0.,4.)]))
        def inner(r,p):
            if r<=0:
                return 0.0
            logvalue=math.log(r)+beta*(r*r/2-r**3/3)-c*(r-m)**2/2-peak
            return math.exp(logvalue)*r**p
        vals=[]
        for power in (0,2):
            vals.append(sum(quad(inner,a,b,args=(power,),epsabs=eps*.01,epsrel=eps*.01)[0]
                            for a,b in zip(splits[:-1],splits[1:])))
        return peak+math.log(vals[0]), vals[1]/vals[0]
    opt=minimize_scalar(lambda m:-conditional(float(m))[0],bounds=(0,rmax),method='bounded',options={'xatol':1e-10})
    outer_mode=float(opt.x)
    max_logj=conditional(outer_mode)[0]
    scale=math.sqrt(1+1/c)/math.sqrt(N)
    def outer(z,p):
        logj,second=conditional(outer_mode+scale*z)
        exponent=N*(logj-max_logj)
        return math.exp(exponent)*(second if p else 1.0)
    integrals=[]
    for power in (0,2):
        value = quad(outer,-width,width,args=(power,),points=[0.],epsabs=eps,epsrel=eps,limit=200)[0]
        integrals.append(value)
    second=integrals[1]/integrals[0]
    return second/_benchmark_free_second(beta,eps=eps*.01,rmax=rmax)


@njit(cache=True, parallel=True)
def _simulate(number, beta, k, omega, dt, burn_steps, measure_steps, seeds, trace_stride=0):
    """One complete N-oscillator realization per independent RNG stream."""
    nrep = len(seeds)
    raw = np.full((nrep, 2), np.nan)
    flags = np.zeros(nrep, dtype=np.int64)
    trace = np.empty((nrep, measure_steps//trace_stride+1 if trace_stride else 0))
    noise = math.sqrt(2*dt/beta)
    for rep in prange(nrep):
        np.random.seed(seeds[rep])
        x, y = np.empty(number), np.empty(number)
        xn, yn = np.empty(number), np.empty(number)
        xp, yp = np.empty(number), np.empty(number)
        fx, fy = np.empty(number), np.empty(number)
        nx, ny = np.empty(number), np.empty(number)
        predicted_r = np.empty(number)
        radius, midpoint_r = np.empty(number), np.empty(number)
        for i in range(number):
            angle = 2*math.pi*np.random.random()
            x[i], y[i] = math.cos(angle), math.sin(angle)
        work = second = 0.0
        for step in range(burn_steps+measure_steps):
            radius_sum = 0.0
            for i in range(number):
                radius[i] = math.hypot(x[i], y[i])
                radius_sum += radius[i]
                if radius[i] == 0.0:
                    flags[rep] = 1
            if flags[rep]:
                break
            if step == burn_steps and trace_stride:
                trace[rep, 0] = x[0]
            predicted_sum = 0.0
            for i in range(number):
                a = 1-radius[i]+k*(radius_sum-number*radius[i])/radius[i]
                fx[i], fy[i] = a*x[i]+omega*y[i], a*y[i]-omega*x[i]
                nx[i], ny[i] = noise*np.random.normal(), noise*np.random.normal()
                xp[i], yp[i] = x[i]+dt*fx[i]+nx[i], y[i]+dt*fy[i]+ny[i]
                predicted_r[i] = math.hypot(xp[i], yp[i])
                predicted_sum += predicted_r[i]
                if predicted_r[i] == 0.0:
                    flags[rep] = 1
            if flags[rep]:
                break
            midpoint_sum = 0.0
            for i in range(number):
                ap = 1-predicted_r[i]+k*(predicted_sum-number*predicted_r[i])/predicted_r[i]
                xn[i] = x[i]+.5*dt*(fx[i]+ap*xp[i]+omega*yp[i])+nx[i]
                yn[i] = y[i]+.5*dt*(fy[i]+ap*yp[i]-omega*xp[i])+ny[i]
                if not (math.isfinite(xn[i]) and math.isfinite(yn[i])):
                    flags[rep] = 2
                if step >= burn_steps:
                    midpoint_r[i] = math.hypot((x[i]+xn[i])/2, (y[i]+yn[i])/2)
                    midpoint_sum += midpoint_r[i]
                    if midpoint_r[i] == 0.0:
                        flags[rep] = 1
            if flags[rep]:
                break
            if step >= burn_steps:
                for i in range(number):
                    xm, ym = (x[i]+xn[i])/2, (y[i]+yn[i])/2
                    dx, dy = xn[i]-x[i], yn[i]-y[i]
                    a = 1-midpoint_r[i]+k*(midpoint_sum-number*midpoint_r[i])/midpoint_r[i]
                    rotation = omega*(ym*dx-xm*dy)
                    work += rotation+a*(xm*dx+ym*dy)
                    rn = math.hypot(xn[i], yn[i])
                    second += (radius[i]**2+rn**2)/2
                measured = step-burn_steps+1
                if trace_stride and measured % trace_stride == 0:
                    trace[rep, measured//trace_stride] = xn[0]
            for i in range(number):
                x[i], y[i] = xn[i], yn[i]
        if flags[rep] == 0:
            duration = measure_steps*dt
            raw[rep, 0] = beta*work/duration
            raw[rep, 1] = beta*omega**2*second/measure_steps
    return raw, flags, trace


def _steps(duration, dt, name, zero=False):
    if not math.isfinite(duration) or duration < 0 or (duration == 0 and not zero):
        raise ValueError(f'{name} must be finite and positive' + (' or zero' if zero else ''))
    count = int(round(duration/dt))
    if count < (0 if zero else 1) or not math.isclose(count*dt, duration, rel_tol=1e-11):
        raise ValueError(f'{name} must be an integer number of timesteps')
    return count


def simulate_replicas(number, beta, k, omega, dt, burn, measure, replicas, seed,
                      trace_interval=None):
    """Cartesian additive-noise Heun; full-drift spatial-midpoint Stratonovich EPR.

    Predictor and corrector reuse the same noise and evaluate the complete
    coupled system synchronously; the EPR midpoint uses accepted states.
    Moments use all measurement steps with trapezoidal time weights. No
    radius clipping, trajectory rejection, or frequency rescaling is applied.
    Seeds exclude replica count, so larger ensembles preserve their prefixes.
    """
    if isinstance(number, bool) or int(number) != number or number < 2:
        raise ValueError('number must be an integer >=2')
    if isinstance(replicas, bool) or int(replicas) != replicas or replicas < 1:
        raise ValueError('replicas must be a positive integer')
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    number, replicas, seed = int(number), int(replicas), int(seed)
    beta, k, omega, dt, burn, measure = map(float, (beta, k, omega, dt, burn, measure))
    if not all(map(math.isfinite, (beta, k, omega, dt))) or beta <= 0 or k < 0 or dt <= 0:
        raise ValueError('Require finite beta>0, k>=0, dt>0 and omega')
    burn_steps, measure_steps = _steps(burn, dt, 'burn', True), _steps(measure, dt, 'measure')
    stride = 0 if trace_interval is None else _steps(float(trace_interval), dt, 'trace_interval')
    if stride and (stride > measure_steps or measure_steps % stride):
        raise ValueError('Trace interval must divide measurement duration')
    settings = dict(number=number, beta=beta, k=k, omega=omega, dt=dt, burn=burn, measure=measure)
    digest = hashlib.blake2b(json.dumps(settings, sort_keys=True).encode(), digest_size=16).digest()
    entropy = [seed, *np.frombuffer(digest, dtype='<u4').tolist()]
    seeds = np.array([s.generate_state(1, dtype=np.uint32)[0]
                      for s in np.random.SeedSequence(entropy).spawn(replicas)], dtype=np.uint32)
    if len(np.unique(seeds)) != replicas:
        raise ValueError('Duplicate replica seeds; choose another master seed')
    raw, flags, trace = _simulate(number, beta, k, omega, dt, burn_steps, measure_steps, seeds, stride)
    if flags.any() or not np.isfinite(raw).all():
        raise FloatingPointError(f'Invalid trajectories: {np.flatnonzero(flags).tolist()}; '
                                 'refine timestep; no trajectories were discarded')
    result = dict(work=raw[:, 0], moment=raw[:, 1], replica_seeds=seeds)
    if stride:
        result.update(trace_x=trace, trace_time=np.arange(trace.shape[1])*stride*dt)
    return result


def interval(values):
    """Pointwise 95% Student interval across independent N-oscillator systems."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite independent observations')
    mean = float(values.mean())
    sem = float(values.std(ddof=1)/math.sqrt(len(values)))
    return dict(mean=mean, sem=sem, half=float(student_t.ppf(.975, len(values)-1)*sem),
                count=len(values))


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
    return {run['id']: {**plan['run_defaults'], **{key: value for key, value in run.items()
                                                  if key != 'id'}}
            for run in plan['runs']}


def verify_checksums(root, checksums):
    for name, digest in checksums.items():
        if checksum(root/'data'/name) != digest:
            raise ValueError(f'Saved data checksum mismatch: {name}')


def read_replicas(root, plan, configs):
    """Load the estimator arrays and seeds from the original or new archives."""
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
    source = Path(__file__).resolve().parent
    plan = load_parameters(source)
    configs = run_configs(plan)

    cases = set(cases or ())
    unknown = cases - configs.keys()
    if unknown:
        raise ValueError('Unknown run IDs: ' + ', '.join(sorted(unknown)))
    runs = {run_id: config for run_id, config in configs.items() if not cases or run_id in cases}
    theory_source = theory_dir or source
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
                boundary_beta=np.array([unity_beta(n) for n in numbers]),
                strong_values=np.array([strong_ratio(.8, n) for n in NUMBERS]))


def calculate_finite_theory():
    """Calculate the finite-coupling stationary EPR curves."""
    beta, numbers = .8, np.array(NUMBERS)
    couplings = np.unique(np.r_[0., np.geomspace(.001, 1., 33),
                               np.linspace(1., 19., 73), .5])
    settings = dict(eps=1e-10, rmax=7., width=12.)
    ratios = np.empty((len(numbers), len(couplings)))
    for i, number in enumerate(numbers):
        print(f'Stationary EPR curve: N={number}, beta={beta:g}', flush=True)
        for j, k in enumerate(couplings):
            ratios[i,j] = stationary_ratio(beta, float(k), int(number), **settings)
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise ArithmeticError('Nonfinite or nonpositive stationary EPR curve')

    return dict(finite_k=couplings, finite_numbers=numbers, finite_ratio=ratios,
                finite_beta=np.asarray(beta))


def numerical_checks(theory):
    """Check the finite-coupling EPR and strong-coupling expression."""
    rows = []
    for i, number in enumerate(theory['finite_numbers']):
        for k in (.001, .5, 19.):
            j = int(np.flatnonzero(theory['finite_k']==k)[0])
            baseline = float(theory['finite_ratio'][i,j])
            beta = float(theory['finite_beta'])
            refined = stationary_ratio(beta, k, int(number), eps=1e-11, rmax=8., width=16.)
            rows.append(dict(check='quadrature', number=int(number), beta=beta, k=k,
                comparison='extend_domains_and_tighten_tolerance', quantity='normalized_epr',
                baseline=baseline, refined=refined, absolute_difference=abs(refined-baseline)))
    max_change = max(row['absolute_difference'] for row in rows)
    if max_change > 1e-8:
        raise ArithmeticError(f'Stationary EPR refinement changed a curve value by {max_change:g}')
    for number in (2, 16):
        for beta in (.1, 10.):
            baseline, refined = (strong_ratio(beta, number, dps=digits) for digits in (35, 70))
            rows.append(dict(check='strong_limit', number=number, beta=beta, k='',
                comparison='double_decimal_digits', quantity='normalized_epr',
                baseline=baseline, refined=refined, absolute_difference=abs(refined-baseline)))
    return rows


def write_theory(root):
    """Save the theory curves and their numerical comparisons."""
    theory = {**calculate_theory(), **calculate_finite_theory()}
    rows = numerical_checks(theory)
    save_npz(root/'data/theory.npz', theory)
    with (root/'data/numerics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    """Replay panel (a) estimator benchmarks and simulation checks."""
    plan = load_parameters(root)
    if 'completed_runs' in plan:
        raise ValueError('Validation requires the complete simulation plan')
    configs = run_configs(plan)
    archive = read_replicas(root, plan, configs)
    verify_checksums(root, {'theory.npz': plan['checksums']['theory.npz']})
    theory = load_npz(root/'data/theory.npz')
    finite_beta = float(theory['finite_beta'])
    quadrature = {(int(number), float(k)): float(theory['finite_ratio'][i, j])
                  for i, number in enumerate(theory['finite_numbers'])
                  for j, k in enumerate(theory['finite_k'])}

    estimators = {'work': 'stratonovich_epr', 'moment': 'radial_moment_epr'}
    normalization, estimates, benchmarks = {}, {}, {}
    for run_id, config in configs.items():
        number, beta, k = config['number'], config['beta'], config['k']
        if beta != finite_beta or (number, k) not in quadrature:
            raise ValueError(f'No stationary reference for {run_id}: N={number}, beta={beta}, k={k}')
        norm = number*beta*config['omega']**2*free_second(beta)
        normalization[run_id] = norm
        work, moment = archive[run_id+'_work'], archive[run_id+'_moment']
        estimates[run_id] = {'work': interval(work/norm), 'moment': interval(moment/norm)}
        row = dict(run_id=run_id, number=number, k=k, quadrature_epr=quadrature[number, k])
        for field, name in estimators.items():
            row.update(estimate_fields(name, estimates[run_id][field]))
        row.update(estimate_fields('stratonovich_minus_radial_moment', interval((work-moment)/norm)))
        benchmarks[run_id] = row

    refinements = []
    for check in plan['refinements']:
        baseline, refined = check['baseline_run'], check['refined_run']
        config = configs[baseline]
        row = dict(changed=check['changed'], baseline_run=baseline, refined_run=refined,
                   number=config['number'], k=config['k'])
        for field, name in estimators.items():
            change = welch_change(estimates[baseline][field], estimates[refined][field])
            row.update(estimate_fields(name+'_change', change))
        refinements.append(row)

    replica_counts = []
    for check in plan['replica_counts']:
        run_id = check['run_id']
        config = configs[run_id]
        for count in check['replicas']:
            row = dict(run_id=run_id, number=config['number'], k=config['k'], replicas=count)
            for field, name in estimators.items():
                estimate = interval(archive[run_id+'_'+field][:count]/normalization[run_id])
                row.update(estimate_fields(name, estimate))
            replica_counts.append(row)

    validation = dict(
        normalization='EPR estimates and differences are divided by the total uncoupled EPR, N sigma_0.',
        uncertainty='Pointwise 95% Student intervals across independent oscillator populations. '
            'Stratonovich and radial-moment estimates share paths; their differences are paired. '
            'Refinement changes are refined minus baseline with Welch intervals for independent runs. '
            'Replica-count estimates use nested subsets.',
        estimator_benchmarks=list(benchmarks.values()),
        timestep_burn_duration_refinements=refinements,
        replica_count_checks=replica_counts)
    plot_fields = ('number', 'k', 'stratonovich_epr', 'stratonovich_epr_sem',
                   'radial_moment_epr', 'radial_moment_epr_sem')
    figure = {'rows': [{field: benchmarks[run_id][field] for field in plot_fields}
                       for run_id in plan['figure_runs']['run_ids']]}
    write_json(root/'data/validation.json', validation)
    write_json(root/'data/figure.json', figure)
    print(f'Saved {len(benchmarks)} simulation summaries and {len(refinements)} refinement comparisons.',
          flush=True)


def publication_style(plt):
    return plt.rc_context({'text.usetex': True, 'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'], 'font.size': 9,
        'axes.labelsize': 11.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'legend.fontsize': 9, 'legend.framealpha': 1., 'axes.linewidth': .8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 4, 'ytick.major.size': 4,
        'axes.grid': False,
        'savefig.bbox': 'tight', 'savefig.transparent': False, 'savefig.facecolor': 'white'})


def save_figure(fig, root, name):
    out = root/'figures'
    out.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'{name}.{extension}', dpi=300)


def plot(root):
    """No simulation, theory evaluation, statistics, or quadrature during plotting."""
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'energetics-figure3-matplotlib'))
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
        label_positions = ((15.5, 5), (15.5, 5), (15.5, 0), (15.5, -5))
        for i, number in enumerate(NUMBERS):
            index = int(np.flatnonzero(theory['finite_numbers']==number)[0])
            ax.plot(theory['finite_k'], theory['finite_ratio'][index], color=colors[i],
                    lw=1.6, zorder=3)
            label_k, label_offset = label_positions[i]
            label_ratio = np.interp(label_k, theory['finite_k'], theory['finite_ratio'][index])
            if number == 8:
                label_ratio = .5*(label_ratio+theory['strong_values'][1])
            ax.annotate(rf'$N={number}$', (label_k, label_ratio),
                        xytext=(0, label_offset), textcoords='offset points',
                        fontsize=11.5, color='black', ha='center',
                        va='center' if number == 8 else ('bottom' if label_offset > 0 else 'top'))
            rows = sorted((r for r in figure['rows'] if r['number']==number), key=lambda r:r['k'])
            for estimator, marker, offset in (('stratonovich_epr', 'x', -.035),
                                               ('radial_moment_epr', 'o', .035)):
                ax.errorbar([r['k']+offset for r in rows],
                    [r[estimator] for r in rows], yerr=[r[estimator+'_sem'] for r in rows],
                    fmt=marker, mfc='white', ms=3.5, color=colors[i], capsize=2,
                    lw=1.2, elinewidth=.9, linestyle='none')
            ax.axhline(theory['strong_values'][i], color=colors[i], ls='--', lw=1.25)
        ax.axhline(1., color='.5', lw=.6, zorder=0)
        ax.set(xlabel=r'$k\;(\tau_r^{-1})$', ylabel=r'$\sigma\;(N\sigma_0)$', xlim=(-.35, 19.))
        handles = [Line2D([], [], color='black', lw=1.6, label='theory'),
                   Line2D([], [], color='.2', marker='o', mfc='white', ls='none', label='radial moments'),
                   Line2D([], [], color='.2', marker='x', ls='none', label='Stratonovich')]
        ax.legend(handles=handles, ncol=1, loc='center', bbox_to_anchor=(.57, .75),
                  fontsize=11.5, frameon=False, handlelength=1.2)
        ax.text(.975, .975, '(a)', transform=ax.transAxes,
                fontsize=12.5, color='black', ha='right', va='top')
        ax.text(.5, .975, rf'$\beta={float(theory["finite_beta"]):g}$',
                transform=ax.transAxes, fontsize=12.5, ha='center', va='top')
        grid = theory['ratio']
        # Match the original manuscript contour's linear colour normalization.
        contour = contour_ax.contourf(theory['number'], theory['beta'], grid,
                                      levels=30, cmap='coolwarm')
        contour_ax.plot(theory['number'], theory['boundary_beta'], 'w-.', lw=1.6)
        boundary_n = 4.8
        boundary_beta = np.interp(boundary_n, theory['number'], theory['boundary_beta'])
        contour_ax.annotate(r'$\beta_1^*$', (boundary_n, boundary_beta),
                            xytext=(0, -7), textcoords='offset points',
                            fontsize=15, color='white', ha='center', va='top')
        # Approximate guide to weak N dependence over the displayed sizes.
        contour_ax.axhline(3., color='.15', ls=':', lw=1.2)
        contour_ax.set(xlabel=r'$N$', ylabel=r'$\beta$', xlim=(2,6), ylim=(.1,10))
        contour_ax.set_xticks([2,3,4,5,6])
        contour_ax.text(.965, .035, '(b)', transform=contour_ax.transAxes,
                        fontsize=12.5, color='black', ha='right', va='bottom')
        inset = contour_ax.inset_axes([.10, .665, .64, .24])
        inset.text(.5, 1.02,
                   rf'$\beta={trace_settings["beta"]:g},\quad '
                   rf'k={trace_settings["k"]:g}\;\tau_r^{{-1}}$',
                   transform=inset.transAxes, fontsize=12.5, ha='center', va='bottom')
        for number, color in ((2, 'lightcoral'), (50, '#377eb8')):
            inset.plot(traces['time'], traces[f'x_{number}'], color=color, lw=1.6)
            inset.annotate(rf'$N={number}$',
                           (traces['time'][-1], traces[f'x_{number}'][-1]),
                           xytext=(-3, 6 if number == 2 else -6),
                           textcoords='offset points', fontsize=12.5,
                           color=color, ha='right',
                           va='bottom' if number == 2 else 'top')
        inset.set(xlim=(0,20), ylim=(-3.1,3.1), xticks=[0,10,20], yticks=[0],
                  xlabel=r'$t\;(\tau_r)$', ylabel=r'$x\;(r_c)$')
        inset.tick_params(labelsize=10.5, length=3, pad=1)
        inset.xaxis.label.set_size(12.5)
        inset.xaxis.labelpad = 0
        inset.yaxis.label.set_size(12.5)
        inset.yaxis.set_label_position('right')
        inset.yaxis.tick_right()
        colorbar = fig.colorbar(contour, ax=contour_ax, fraction=.055, pad=.025,
                               ticks=[.85,.9,.95,1.])
        colorbar.ax.axhline(1., color='white', ls=(0, (1.5, 1.2)), lw=1.6)
        colorbar.set_label(r'$\sigma(k\to\infty)\;(N\sigma_0)$', fontsize=13)
        for panel in (ax, contour_ax):
            panel.xaxis.label.set_size(13)
            panel.yaxis.label.set_size(13)
        save_figure(fig, root, 'figure3')
        plt.close(fig)
    print(f'Saved PNG and SVG figures in {root/"figures"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="plot", choices=("plot", "compute", "validate", "theory"))
    parser.add_argument("--outdir", type=Path, help="Data directory; compute requires a separate, empty directory")
    parser.add_argument("--threads", type=int, help="Numba threads during simulation")
    parser.add_argument("--case", action="append", help="Compute only this recorded run ID (repeatable)")

    parser.add_argument("--theory-dir", type=Path, help="Use theory and numerical checks from this calculation directory")
    args = parser.parse_args(argv)
    root = (args.outdir or Path(__file__).resolve().parent).resolve()
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
        write_theory(root)
    else:
        plot(root)


if __name__=='__main__':
    main()
