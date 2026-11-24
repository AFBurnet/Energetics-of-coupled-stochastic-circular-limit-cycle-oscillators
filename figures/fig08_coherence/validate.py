"""Replay Figure 8 endpoint and uncertainty checks from saved bootstrap draws."""
import csv
import json

import numpy as np


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True):
        raise ValueError(f"{label} differs from its recorded evidence.")


def ci(draws):
    return np.quantile(draws, [.025, .975])


def case_key(row):
    return row["case_key"], row["grid"], row["readout"]


def endpoint_draws(row, arrays):
    return arrays[row["bootstrap_key"]][:, int(row["bootstrap_column"])]


def validate(here):
    data, evidence = here/"data", here/"validation"
    settings = json.loads((data/"settings.json").read_text())
    cases = {row["case_key"]: row for row in rows(data/"cases.csv")}
    endpoints = rows(evidence/"endpoint_counts.csv")
    counts = {case_key(row): row for row in endpoints}
    if len(counts) != len(endpoints):
        raise ValueError("Duplicate endpoint evidence.")
    with np.load(evidence/"endpoint_bootstrap.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for row in endpoints:
        case = cases[row["case_key"]]
        draws = endpoint_draws(row, arrays)
        if len(draws) != settings[case["study"]]["bootstrap"]:
            raise ValueError("Bootstrap evidence has the wrong sample count.")
        if not np.isfinite(draws).all():
            raise ValueError(f"Some saved endpoint draws are unresolved: {case_key(row)}")
        check(float(row["endpoint_cycles"]), abs(float(row["phase_at_coherence"]))/(2*np.pi),
              "Coherence count")
        check(ci(draws), [float(row["endpoint_ci_low"]), float(row["endpoint_ci_high"])],
              "Endpoint interval")

    plotted = rows(data/"figure.csv")
    with np.load(evidence/"ratio_bootstrap.npz", allow_pickle=False) as ratio_arrays:
        keys = [row["bootstrap_key"] for row in plotted if row["bootstrap_key"]]
        if len(keys) != len(set(keys)) or set(keys) != set(ratio_arrays.files):
            raise ValueError("Plot table and ratio bootstrap records disagree.")
        for row in plotted:
            value = float(row["value"])
            panel = row["panel"]
            if panel in "bcf":
                record = counts[row["case_key"], "reference", row["readout"]]
                draws = endpoint_draws(record, arrays)
                check(value, float(record["endpoint_cycles"]), "Plotted endpoint")
            elif row["coupling"] == "shared_uncoupled" or panel == "e" and row["condition"] == "uncoupled":
                draws = np.ones(2)
                check(value, 1., "Self-normalized baseline")
            else:
                draws = ratio_arrays[row["bootstrap_key"]]
                if len(draws) != settings[cases[row["case_key"]]["study"]]["bootstrap"]:
                    raise ValueError("Ratio evidence has the wrong sample count.")
            check([float(row[key]) for key in ("ci_low", "ci_high")], ci(draws), "Plotted interval")
            check(float(row["standard_error"]), np.std(draws, ddof=1), "Plotted standard error")

    comparison_count = 0
    for name in ("endpoint_contrasts.csv", "endpoint_refinements.csv"):
        for row in rows(evidence/name):
            counter = counts[case_key(row)]
            base = counts[row["reference_case_key"], row["reference_grid"], row["readout"]]
            difference = endpoint_draws(counter, arrays) - endpoint_draws(base, arrays)
            result = float(counter["endpoint_cycles"]) - float(base["endpoint_cycles"])
            check([float(row[key]) for key in ("endpoint_difference", "ci_low", "ci_high")],
                  [result, *ci(difference)], name)
            comparison_count += 1
    benchmarks = rows(evidence/"epr_benchmarks.csv")
    for row in benchmarks:
        check(float(row["difference"]),
              float(row["stratonovich_epr"]) - float(row["radial_moment_epr"]),
              "EPR estimator difference")
    print(f"Verified uncertainty for {len(plotted)} plotted points, {len(endpoints)} saved endpoints, "
          f"{comparison_count} contrast and refinement intervals, and {len(benchmarks)} EPR differences.")


def validate_calculation(root):
    """Check new bootstrap summaries and rebuild comparisons from available cases."""
    from compute import CYCLES, RADIAL_MOMENT, RMS, STRATONOVICH, CASE_FIELDS, TARGETED_TIMESTEPS, write_csv

    measurements = rows(root/"measurements.csv")
    plotted = rows(root/"figure.csv")
    metadata = json.loads((root/"cases.json").read_text())
    with np.load(root/"bootstrap_draws.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    indexed = {(row["case_key"], row["grid"], row["metric"], row["readout"]): row
               for row in measurements}
    if len(indexed) != len(measurements):
        raise ValueError("Duplicate calculation estimates.")
    for row in (*measurements, *plotted):
        draws = arrays[row["bootstrap_key"]]
        if draws.shape != (metadata[row["case_key"]]["bootstrap_count"],):
            raise ValueError("Bootstrap evidence has the wrong sample count.")
        check(ci(draws), [float(row["ci_low"]), float(row["ci_high"])], "Interval")
        check(np.std(draws, ddof=1), float(row["standard_error"]), "Standard error")
    metrics = {"epr_ratio": STRATONOVICH, "rms_amplitude_ratio": RMS,
               "signal_n_coherent_cycles": CYCLES}
    for row in plotted:
        key = (row["grid"], metrics[row["metric"]], row["readout"])
        case, reference = row["case_key"], row["reference_case_key"]
        estimate = indexed[(case, *key)]
        value, draws = float(estimate["value"]), arrays[estimate["bootstrap_key"]]
        if row["panel"] in "ade":
            ref = indexed[(reference, *key)]
            check(float(row["baseline_value"]), float(ref["value"]), "Ratio baseline")
            if case == reference:
                value, draws = 1., np.ones_like(draws)
            else:
                value, draws = value/float(ref["value"]), draws/arrays[ref["bootstrap_key"]]
        check(value, float(row["value"]), "Plotted value")
        check(draws, arrays[row["bootstrap_key"]], "Plotted bootstrap draws")

    def context(row):
        return {key: row[key] for key in CASE_FIELDS}

    def physical_case(row, *, uncoupled=False):
        return (row["study"], row["population_N"], row["coupling"] if not uncoupled else "",
                float(row["k"]) if not uncoupled else 0.)

    references = {physical_case(row, uncoupled=float(row["k"]) == 0): row["case_key"]
                  for row in measurements if row["variant"] == "reference"}

    half_dt = {physical_case(row): row["case_key"] for row in measurements if row["variant"] == "half_dt"}

    def reference_row(row, *, uncoupled=False, grid=None, case=None):
        if case is None:
            case = references.get(physical_case(row, uncoupled=uncoupled or float(row["k"]) == 0))
        return indexed.get((case, grid or row["grid"], row["metric"], row["readout"]))

    def difference(row, ref, field):
        value = float(row["value"])-float(ref["value"])
        draws = arrays[row["bootstrap_key"]]-arrays[ref["bootstrap_key"]]
        low, high = ci(draws)
        return {field: value, "ci_low": low, "ci_high": high,
                "reference_case_key": ref["case_key"]}

    def endpoint_difference(row, ref, pairing):
        return dict(context(row), grid=row["grid"], readout=row["readout"], pairing=pairing,
                    reference_grid=ref["grid"],
                    plot_included=row["plot_included"] == ref["plot_included"] == "True",
                    **difference(row, ref, "endpoint_difference"))

    endpoints, contrasts, refinements, changes, benchmarks = [], [], [], [], []
    for row in measurements:
        metric, study = row["metric"], row["study"]
        variant, grid, readout, case = (row[key] for key in ("variant", "grid", "readout", "case_key"))
        if metric == CYCLES:
            check(float(row["value"]), abs(float(row["phase_at_coherence"]))/(2*np.pi), "Coherence count")
            endpoint = dict(context(row), grid=grid, readout=readout, endpoint_cycles=row["value"],
                            endpoint_ci_low=row["ci_low"], endpoint_ci_high=row["ci_high"],
                            tau_coherence=row["tau_coherence"], phase_at_coherence=row["phase_at_coherence"],
                            bootstrap_key=row["bootstrap_key"], plot_included=row["plot_included"])
            endpoints.append(endpoint)
            if grid == "sampling_base" or (variant == "sampling" and study == "pair"):
                continue
            if variant in TARGETED_TIMESTEPS:
                if variant == "quarter_dt" and grid == "reference":
                    uncoupled = reference_row(row, uncoupled=True)
                    if uncoupled is not None:
                        contrasts.append(endpoint_difference(row, uncoupled, "independent_runs"))
                    half = half_dt.get(physical_case(row))
                    if half is not None:
                        ref = reference_row(row, case=half)
                        refinements.append(endpoint_difference(row, ref, "independent_runs"))
                continue
            contrast = grid == "reference" and variant == "reference"
            matched = grid == "fine_lags" or variant == "sampling"
            if grid == "fine_lags":
                ref = reference_row(row, case=case, grid="reference")
            elif variant == "sampling":
                ref = reference_row(row, case=case, grid="sampling_base")
            elif contrast and float(row["k"]) == 0:
                continue
            else:
                ref = reference_row(row, uncoupled=contrast)
            if ref is None:
                continue
            target = contrasts if contrast else refinements
            target.append(endpoint_difference(row, ref, "matched_paths" if matched else "independent_runs"))
        elif metric in (STRATONOVICH, RMS) and variant in ("coarse", "fine", "burn", "duration"):
            ref = reference_row(row)
            if ref is not None:
                changes.append(dict(context(row), metric=metric, readout=readout,
                                    **difference(row, ref, "difference")))
        elif metric == STRATONOVICH and study == "population" and variant in ("reference", *TARGETED_TIMESTEPS):
            if variant == "quarter_dt":
                half = half_dt.get(physical_case(row))
                if half is not None:
                    changes.append(dict(context(row), metric=metric, readout=readout,
                                        **difference(row, reference_row(row, case=half), "difference")))
            ref = indexed[case, grid, RADIAL_MOMENT, readout]
            values = difference(row, ref, "difference")
            del values["reference_case_key"]
            benchmarks.append(dict(population_N=row["population_N"], condition=row["condition"],
                                   case_key=case, stratonovich_epr=row["value"],
                                   stratonovich_ci_low=row["ci_low"], stratonovich_ci_high=row["ci_high"],
                                   radial_moment_epr=ref["value"], **values))

    evidence = root/"validation"
    evidence.mkdir(exist_ok=True)
    endpoint_columns = (*CASE_FIELDS, "grid", "readout", "pairing", "reference_grid", "plot_included",
                        "endpoint_difference", "ci_low", "ci_high", "reference_case_key")
    write_csv(evidence/"endpoint_counts.csv", endpoints)
    write_csv(evidence/"endpoint_contrasts.csv", contrasts, endpoint_columns)
    write_csv(evidence/"endpoint_refinements.csv", refinements, endpoint_columns)
    write_csv(evidence/"epr_amplitude_refinements.csv", changes,
              (*CASE_FIELDS, "metric", "readout", "difference", "ci_low", "ci_high", "reference_case_key"))
    write_csv(evidence/"epr_benchmarks.csv", benchmarks,
              ("population_N", "condition", "case_key", "stratonovich_epr", "stratonovich_ci_low",
               "stratonovich_ci_high", "radial_moment_epr", "difference", "ci_low", "ci_high"))
    print(f"Verified {len(plotted)} figure points and wrote {len(contrasts)} coherence contrasts, "
          f"{len(refinements)} coherence and {len(changes)} EPR/amplitude refinements, "
          f"and {len(benchmarks)} EPR benchmarks.")
