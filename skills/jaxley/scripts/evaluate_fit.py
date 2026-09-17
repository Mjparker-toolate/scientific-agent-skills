#!/usr/bin/env python3
"""Score a (fitted) model against target traces on a protocol.

Simulates the model under ``--protocol`` (ideally a held-out protocol the fit
never saw), compares with ``--target``, and writes a JSON report with per-trial
voltage RMSE/MAE, spike counts, first-spike latencies, and their errors. Exit
status is 1 when ``--max-rmse`` or ``--max-spike-count-error`` is violated, so
the script doubles as a validation gate in a pipeline.

Example::

    python scripts/evaluate_fit.py --model fit/fitted_model.json \
        --protocol assets/heldout_protocol.json --target heldout.npz \
        --out fit/heldout_report.json --max-rmse 5.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as common  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model specification JSON (e.g. fitted_model.json)")
    parser.add_argument("--protocol", required=True, help="protocol JSON")
    parser.add_argument("--target", required=True, help=".npz with target `traces`")
    parser.add_argument("--out", required=True, help="output report JSON")
    parser.add_argument("--save-traces", default=None, help="also save the model traces to this .npz")
    parser.add_argument("--spike-threshold", type=float, default=common.SPIKE_THRESHOLD_MV, help="mV threshold for spike detection (default 0)")
    parser.add_argument("--max-rmse", type=float, default=None, help="fail when the mean voltage RMSE (mV) exceeds this")
    parser.add_argument("--max-spike-count-error", type=float, default=None, help="fail when the mean absolute spike-count error exceeds this")
    parser.add_argument("--no-x64", action="store_true", help="run JAX in float32")
    parser.add_argument("--platform", default=None, help="force a JAX platform, e.g. cpu or gpu")
    return parser.parse_args(argv)


def trial_metrics(prediction: np.ndarray, target: np.ndarray, time: np.ndarray, threshold: float) -> dict:
    """Metrics for one recording of one trial."""
    error = prediction - target
    predicted_spikes = common.spike_times(prediction, time, threshold)
    target_spikes = common.spike_times(target, time, threshold)
    latency_error = None
    if predicted_spikes.size and target_spikes.size:
        latency_error = float(predicted_spikes[0] - target_spikes[0])
    return {
        "rmse_mV": float(np.sqrt(np.mean(error**2))),
        "mae_mV": float(np.mean(np.abs(error))),
        "max_abs_error_mV": float(np.max(np.abs(error))),
        "spikes_model": int(predicted_spikes.size),
        "spikes_target": int(target_spikes.size),
        "spike_count_error": int(predicted_spikes.size - target_spikes.size),
        "first_spike_ms_model": float(predicted_spikes[0]) if predicted_spikes.size else None,
        "first_spike_ms_target": float(target_spikes[0]) if target_spikes.size else None,
        "first_spike_latency_error_ms": latency_error,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common.configure_jax(x64=not args.no_x64, platform=args.platform)

    spec = common.load_json(args.model)
    protocol = common.load_protocol(args.protocol)
    target = common.load_array(args.target, key="traces")
    expected = (protocol.n_trials, len(protocol.recordings), protocol.n_time)
    if tuple(target.shape) != expected:
        print(f"error: target traces have shape {target.shape}, protocol expects {expected}", file=sys.stderr)
        return 2

    cell = common.build_model(spec, base_dir=Path(args.model).resolve().parent)
    simulate = common.jax.jit(common.make_simulator(cell, protocol))
    prediction = np.asarray(simulate([], common.jnp.asarray(protocol.currents)))

    trials = []
    for trial_index, label in enumerate(protocol.labels):
        for recording_index, recording in enumerate(protocol.recordings):
            metrics = trial_metrics(
                prediction[trial_index, recording_index],
                target[trial_index, recording_index],
                protocol.time,
                args.spike_threshold,
            )
            metrics.update({"trial": label, "recording": recording_index, "state": recording.get("state", "v")})
            trials.append(metrics)

    mean_rmse = float(np.mean([trial["rmse_mV"] for trial in trials]))
    mean_spike_error = float(np.mean([abs(trial["spike_count_error"]) for trial in trials]))
    latencies = [trial["first_spike_latency_error_ms"] for trial in trials if trial["first_spike_latency_error_ms"] is not None]
    failures = []
    if args.max_rmse is not None and mean_rmse > args.max_rmse:
        failures.append(f"mean RMSE {mean_rmse:.3f} mV > {args.max_rmse}")
    if args.max_spike_count_error is not None and mean_spike_error > args.max_spike_count_error:
        failures.append(f"mean |spike-count error| {mean_spike_error:.3f} > {args.max_spike_count_error}")

    report = {
        "model": str(Path(args.model).resolve()),
        "protocol": str(Path(args.protocol).resolve()),
        "target": str(Path(args.target).resolve()),
        "n_trials": protocol.n_trials,
        "n_recordings": len(protocol.recordings),
        "mean_rmse_mV": mean_rmse,
        "mean_abs_spike_count_error": mean_spike_error,
        "mean_abs_first_spike_latency_error_ms": float(np.mean(np.abs(latencies))) if latencies else None,
        "passed": not failures,
        "failures": failures,
        "trials": trials,
    }
    common.dump_json(report, args.out)
    if args.save_traces:
        Path(args.save_traces).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_traces,
            traces=prediction,
            time=protocol.time,
            currents=protocol.currents,
            labels=np.asarray(protocol.labels),
        )

    print(f"{'trial':<22}{'rec':>4}{'RMSE mV':>10}{'spikes':>14}{'latency err ms':>16}")
    for trial in trials:
        latency = trial["first_spike_latency_error_ms"]
        print(
            f"{trial['trial']:<22}{trial['recording']:>4}{trial['rmse_mV']:>10.3f}"
            f"{trial['spikes_model']:>7}/{trial['spikes_target']:<6}"
            f"{'-' if latency is None else f'{latency:.3f}':>16}"
        )
    print(f"mean RMSE {mean_rmse:.3f} mV, mean |spike-count error| {mean_spike_error:.3f}")
    for failure in failures:
        print(f"FAIL: {failure}")
    print(f"wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
