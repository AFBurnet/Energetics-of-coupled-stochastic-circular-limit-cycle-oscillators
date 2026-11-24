"""Simulate Figure 8 cases and calculate coherence, EPR and amplitude uncertainties."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from endpoint import endpoint_metrics

READOUTS = ("oscillator_1", "collective")
ENDPOINT = "phase_at_first_1_over_e_crossing"
STRATONOVICH = "stratonovich_epr"
RADIAL_MOMENT = "radial_moment_epr"
CYCLES = "endpoint_cycles"
RMS = "rms_amplitude"
CASE_FIELDS = ("study", "variant", "coupling", "k", "population_N", "condition", "case_key")
TARGETED_TIMESTEPS = ("half_dt", "quarter_dt")


def add_compute_arguments(parser):
    parser.add_argument("--study", choices=("pair", "population", "both"), required=True)
    parser.add_argument("--outdir", type=Path, required=True, help="New output directory")
    parser.add_argument("--couplings", nargs="+", help="Select pair coupling types")
    parser.add_argument("--k", type=float, nargs="+", help="Select nonzero pair weights; k=0 is always run")
    parser.add_argument("--sizes", type=int, nargs="+", help="Select population sizes")
    variants = ("reference", "coarse", "fine", "burn", "duration", "sampling", *TARGETED_TIMESTEPS, "all")
    parser.add_argument("--variant", choices=variants, default="reference")
    parser.add_argument("--lag-grid", choices=("reference", "fine_lags", "sampling_base"),
                        help="Select a population lag grid (default: all recorded grids)")
    for name in ("replicas", "bootstrap", "threads"):
        parser.add_argument("--" + name, type=int)
    for name in ("burn", "duration", "dt", "lag-max"):
        parser.add_argument("--" + name, type=float)


def lag_offsets(max_time, saved_spacing, base_spacing, omega, count):
    """Log plus rotation-resolving linear lags, on the original physical grid."""
    maximum = int(np.floor(max_time/base_spacing))
    if maximum < 1:
        raise ValueError("The lag window must span at least one analysis interval.")
    rotation_stride = max(1, int(np.pi/(4*abs(omega)*base_spacing))) if omega else maximum
    stride = max(1, min(maximum//(count-1), rotation_stride))
    grid = np.unique(np.r_[0, np.arange(1, min(9, maximum+1)),
                          np.geomspace(1, maximum, count).round().astype(int),
                          np.arange(0, maximum+1, stride), maximum])
    times = grid*base_spacing
    lags = np.rint(times/saved_spacing).astype(int)
    if not np.allclose(lags*saved_spacing, times, atol=1e-10, rtol=0):
        raise ValueError("Saving intervals do not divide the analysis grid.")
    return lags


def replica_correlations(signals, lags):
    """Known zero stationary mean; pool time-origin moments before normalizing."""
    from scipy.fft import next_fast_len

    n = signals.shape[1]
    nfft = next_fast_len(2*n-1)
    pieces = []
    for start in range(0, len(signals), 8):
        fourier = np.fft.fft(signals[start:start+8], n=nfft, axis=1)
        sums = np.fft.ifft(fourier*fourier.conj(), axis=1)
        pieces.append(sums[:, lags]/(n-lags)[None, :, None])
    return np.concatenate(pieces), np.mean(np.abs(signals)**2, axis=1)


def coherence(correlation, power, time, weights):
    """Recompute the crossing in each whole-system bootstrap sample."""
    point_power, draws_power = power.mean(axis=0), weights@power
    if np.any(point_power <= 0) or np.any(draws_power <= 0):
        raise ValueError("A zero-power signal does not define a coherence estimate.")
    center = correlation.mean(axis=0)[None]/point_power
    samples = np.stack([(weights@correlation[:, :, j])/draws_power[:, j, None]
                        for j in range(len(READOUTS))], axis=2)
    estimate, draws = endpoint_metrics(center, time), endpoint_metrics(samples, time)
    result = {}
    for j, readout in enumerate(READOUTS):
        point = float(estimate["n_coherent_cycles"][0, j])
        sample = draws["n_coherent_cycles"][:, j]
        included = (np.isfinite(point) and np.isfinite(sample).all()
                    and not estimate["below_sample_resolution"][0, j]
                    and not estimate["phase_resolution_warning"][0, j]
                    and not draws["phase_resolution_warning"][:, j].any())
        details = {key: float(estimate[key][0, j])
                   for key in ("tau_coherence", "phase_at_coherence")}
        result[CYCLES, readout] = (point, sample, included), details
    return result


def run_case(record, setup, args):
    from simulation import simulate

    pair = record["study"] == "pair"
    settings = setup[record["study"]]
    n = int(record["population_N"])
    omegas = settings["omegas"] if pair else [settings["omega"]]*n
    sample_dt = float(record["sample_dt"])
    duration = float(record["duration"]) if args.duration is None else args.duration
    burn = float(record["burn"]) if args.burn is None else args.burn
    dt = float(record["dt_max"]) if args.dt is None else args.dt
    lag_max = float(record["lag_max"]) if args.lag_max is None else args.lag_max
    grids = ["reference"]
    if not pair and record["variant"] not in TARGETED_TIMESTEPS:
        grids.append("fine_lags")
        if record["variant"] == "sampling":
            grids.append("sampling_base")
    if args.lag_grid:
        grids = [args.lag_grid]
    base_grid = settings["sample_dt"]*(2 if pair else 1)
    lags, times = {}, {}
    for grid in grids:
        spacing = sample_dt*(2 if grid == "sampling_base" else 1)
        count = int(settings["lag_count"])
        if not pair:
            limit = settings["max_lag_step"]/(2 if grid == "fine_lags" else 1)
            count = max(count, int(np.ceil(lag_max/limit))+1)
        lags[grid] = lag_offsets(lag_max, spacing, base_grid, max(map(abs, omegas)), count)
        times[grid] = lags[grid]*spacing
        if times[grid][-1] >= np.ceil(duration/sample_dt)*sample_dt:
            raise ValueError("The duration must exceed the analyzed lag window.")
    direct_grids = [grid for grid in grids if grid != "sampling_base"]
    if direct_grids:
        union = np.unique(np.concatenate([lags[grid] for grid in direct_grids]))
        indices = {grid: np.searchsorted(union, lags[grid]) for grid in direct_grids}
    published_counts = list(map(int, record["batch_sizes"].split()))
    replicas = sum(published_counts) if args.replicas is None else args.replicas
    counts = published_counts if args.replicas is None else [
        min(published_counts[0], replicas-i) for i in range(0, replicas, published_counts[0])]
    published_seeds = list(map(int, record["batch_seeds"].split()))
    seeds = [published_seeds[i] if i < len(published_seeds) else
             int.from_bytes(hashlib.sha256(f'{record["case_key"]}/{i}'.encode()).digest()[:4], "little")
             for i in range(len(counts))]
    config = dict(k=float(record["k"]), T_eff=settings["T_eff"], omegas=omegas,
                  coupling=record["coupling"] if pair and float(record["k"]) else "cartesian",
                  burn=burn, duration=duration, dt_max=dt, sample_dt=sample_dt)
    correlations = {grid: [] for grid in grids}
    powers = {grid: [] for grid in grids}
    eprs, radial_eprs, amplitudes, replica_seeds = [], [], [], []
    for count, seed in zip(counts, seeds):
        batch = simulate(**config, n_replicas=count, seed=seed)
        # Group each readout contiguously for the correlation reductions.
        signals = batch["signals"][:, :, [0, 1]]
        if direct_grids:
            corr, power = replica_correlations(signals, union)
            for grid in direct_grids:
                correlations[grid].append(corr[:, indices[grid]])
                powers[grid].append(power)
        if "sampling_base" in grids:
            corr, power = replica_correlations(signals[:, ::2], lags["sampling_base"])
            correlations["sampling_base"].append(corr)
            powers["sampling_base"].append(power)
        eprs.append(batch["epr"])
        if not pair:
            radial_eprs.append(batch[RADIAL_MOMENT])
        amplitudes.append(batch["rms_amplitude"])
        replica_seeds.extend(map(int, batch["seeds"]))
        actual = {key: batch[key] for key in ("dt", "burn", "duration", "sample_dt")}
        del batch, signals
    if len(set(replica_seeds)) != replicas:
        raise ValueError("Duplicate replica seeds.")
    boot_seed = int(record["bootstrap_seed"])
    n_draws = settings["bootstrap"] if args.bootstrap is None else args.bootstrap
    weights = np.random.default_rng(boot_seed).multinomial(
        replicas, np.full(replicas, 1/replicas), size=n_draws)/replicas
    summary = {}
    for grid in grids:
        summary.update({(grid, *key): value for key, value in coherence(
            np.concatenate(correlations[grid]), np.concatenate(powers[grid]), times[grid], weights).items()})
    epr = np.concatenate(eprs)
    summary["simulation", STRATONOVICH, "total_system"] = (float(epr.mean()), weights@epr, True), {}
    if radial_eprs:
        radial_epr = np.concatenate(radial_eprs)
        summary["simulation", RADIAL_MOMENT, "total_system"] = (
            float(radial_epr.mean()), weights@radial_epr, True), {}
    if pair:
        radii_sq = np.concatenate(amplitudes)**2
        for j, readout in enumerate(READOUTS):
            values = radii_sq[:, j]
            summary["simulation", RMS, readout] = (
                float(np.sqrt(values.mean())), np.sqrt(weights@values), True), {}
    meta = dict(settings=config, achieved_times=actual,
                lag_times={grid: value.tolist() for grid, value in times.items()},
                batch_sizes=counts, batch_seeds=seeds, replica_seeds=replica_seeds,
                bootstrap_seed=boot_seed, bootstrap_count=n_draws)
    return summary, meta


def result_fields(result, baseline=None, exact=False):
    value, draws, included = result
    if baseline is not None:
        value = value/baseline[0]
        draws = draws/baseline[1]
    if exact:
        value, draws = 1., np.ones_like(draws)
    if np.isfinite(draws).all():
        low, high = np.quantile(draws, [.025, .975])
        se = float(np.std(draws, ddof=1))
    else:
        low, high, se = np.nan, np.nan, np.nan
    included = bool(included and np.isfinite([value, se]).all())
    return dict(value=value, ci_low=low, ci_high=high, standard_error=se,
                plot_included=included,
                exclusion_reason="" if included else "unresolved_or_insufficient_phase_resolution"), draws


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_records(args, here, setup, couplings):
    studies = ("pair", "population") if args.study == "both" else (args.study,)
    if "population" not in studies and args.sizes or "pair" not in studies and (args.k or args.couplings):
        raise ValueError("Use --k and --couplings for pairs, and --sizes for populations.")
    if args.lag_grid in ("fine_lags", "sampling_base") and studies != ("population",):
        raise ValueError("Alternate lag grids apply only to populations.")
    if args.lag_grid == "sampling_base" and args.variant != "sampling":
        raise ValueError("The decimated lag grid requires the sampling variant.")
    pair_couplings = set(args.couplings or couplings)
    pair_k = set(args.k or setup["pair"]["k"]) - {0.}
    sizes = set(args.sizes or ())
    with (here/"data/cases.csv").open(newline="") as stream:
        recorded = [row for row in csv.DictReader(stream) if row["study"] in studies]
    selected = [row for row in recorded if args.variant in (row["variant"], "all")]
    if "population" in studies and args.variant == "reference":
        published_key = setup["population"].get("published_case")
        if published_key is not None:
            published = next(row for row in recorded if row["case_key"] == published_key)
            selected = plot_records([*selected, published], "reference", published_key)
    if args.variant in TARGETED_TIMESTEPS:
        if not selected:
            raise ValueError("The targeted timestep cases are recorded only for the coupled population.")
        size = selected[0]["population_N"]
        selected += [row for row in recorded if row["study"] == "population"
                     and row["variant"] == "reference" and row["population_N"] == size
                     and row["condition"] == "uncoupled"]
    pair = [r for r in selected if r["study"] == "pair" and (float(r["k"]) == 0 or
            (r["coupling"] in pair_couplings and float(r["k"]) in pair_k))]
    if "pair" in studies:
        available = {(r["coupling"], float(r["k"])) for r in pair}
        if not pair_k or not {(c, k) for c in pair_couplings for k in pair_k} <= available:
            raise ValueError("The selected strengths and couplings must be present in the figure protocol.")
    population = [r for r in selected if r["study"] == "population" and
                  (not sizes or int(r["population_N"]) in sizes)]
    if "population" in studies and (not population or sizes - {int(r["population_N"]) for r in population}):
        raise ValueError("The selected sizes must be present in the figure protocol for this variant.")
    return pair + population


def plot_records(records, variant, published_key):
    """Use the finer published point without changing the refinement reference."""
    plotted = [row for row in records if row["variant"] == variant]
    if variant in TARGETED_TIMESTEPS:
        plotted += [row for row in records if row["variant"] == "reference"]
    elif variant == "reference" and published_key:
        published = next((row for row in records if row["case_key"] == published_key), None)
        if published is not None:
            plotted = [row for row in plotted if not (
                row["study"] == "population" and row["condition"] == "coupled" and
                row["population_N"] == published["population_N"])]
            plotted.append(published)
    return plotted


def compute(args, here, couplings):
    from numba import config, set_num_threads

    setup = json.loads((here/"data/settings.json").read_text())
    if (args.replicas is not None and args.replicas < 2
            or args.bootstrap is not None and args.bootstrap < 20
            or args.threads is not None and args.threads < 1):
        raise ValueError("Need at least 2 replicas, 20 bootstrap draws and a positive thread count.")
    set_num_threads(min(args.threads if args.threads is not None else 4, config.NUMBA_NUM_THREADS))
    records = selected_records(args, here, setup, couplings)
    args.outdir.mkdir(parents=True, exist_ok=False)
    summaries, metadata, measurements, rows, arrays = {}, {}, [], [], {}
    for r in records:
        print(f"{r['study']} {r['coupling']} N={r['population_N']} k={r['k']} "
              f"{r['condition']} {r['variant']}", flush=True)
        case = r["case_key"]
        summaries[case], metadata[case] = run_case(r, setup, args)
        for (grid, metric, readout), (result, details) in summaries[case].items():
            fields, draws = result_fields(result)
            bootstrap_key = "__".join((case, grid, metric, readout))
            arrays[bootstrap_key] = draws
            measurements.append(dict(**{key: r[key] for key in CASE_FIELDS}, grid=grid,
                                     metric=metric, readout=readout, bootstrap_key=bootstrap_key,
                                     tau_coherence=details.get("tau_coherence", ""),
                                     phase_at_coherence=details.get("phase_at_coherence", ""), **fields))
    variant = "reference" if args.variant == "all" else args.variant
    plot_cases = plot_records(records, variant, setup["population"].get("published_case"))
    baselines = {(r["study"], r["population_N"]): r["case_key"] for r in plot_cases if float(r["k"]) == 0}
    for r in plot_cases:
        pair = r["study"] == "pair"
        n, case = r["population_N"], r["case_key"]
        reference_case = baselines[r["study"], n]
        values, reference = summaries[case], summaries[reference_case]
        exact = case == reference_case
        coupling = ("shared_uncoupled" if exact else r["coupling"]) if pair else (
            "uncoupled" if exact else "cartesian")
        metrics = [("a" if pair else "e", "epr_ratio", ("simulation", STRATONOVICH, "total_system"))]
        for readout, panel in zip(READOUTS, "bc" if pair else "ff"):
            metrics.append((panel, "signal_n_coherent_cycles", (args.lag_grid or "reference", CYCLES, readout)))
            if pair:
                metrics.append(("d", "rms_amplitude_ratio", ("simulation", RMS, readout)))
        for panel, metric, key in metrics:
            ratio = panel in "ade"
            fields, draws = result_fields(values[key][0], reference[key][0] if ratio and not exact else None,
                                          exact=ratio and exact)
            bootstrap_key = "__".join((case, "ratio" if ratio else key[0], *key[1:]))
            arrays[bootstrap_key] = draws
            rows.append(dict(panel=panel, coupling=coupling, k=r["k"] if pair else "",
                             readout=key[2], metric=metric, population_N=n, condition=r["condition"],
                             K="" if pair else setup["population"]["K"], k_edge=r["k"], case_key=case,
                             reference_case_key=reference_case, grid=key[0], bootstrap_key=bootstrap_key,
                             variant=r["variant"], coherence_definition="" if ratio else ENDPOINT,
                             baseline_value=reference[key][0][0], **fields))
    write_csv(args.outdir/"figure.csv", rows)
    write_csv(args.outdir/"measurements.csv", measurements)
    np.savez_compressed(args.outdir/"bootstrap_draws.npz", **arrays)
    (args.outdir/"cases.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Saved {len(rows)} figure points and {len(measurements)} bootstrap estimates to {args.outdir}.")
