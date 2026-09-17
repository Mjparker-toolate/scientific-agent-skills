#!/usr/bin/env python3
"""Simulate a JSON model specification under a stimulation protocol.

Writes an ``.npz`` with ``traces`` of shape ``(n_trials, n_recordings, n_time)``
(mV), ``time`` (ms), ``currents`` (nA), and trial ``labels``. The same file
format is what ``fit_biophysics.py`` fits to and ``evaluate_fit.py`` scores
against, so this script produces synthetic ground-truth targets and model
predictions alike.

Example::

    python scripts/simulate_protocol.py --model assets/hh_soma.json \
        --protocol assets/step_protocol.json --out target.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as common  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model specification JSON")
    parser.add_argument("--protocol", required=True, help="protocol JSON (steps or currents_file)")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--spike-threshold", type=float, default=common.SPIKE_THRESHOLD_MV, help="mV threshold for the spike-count summary (default 0)")
    parser.add_argument("--no-x64", action="store_true", help="run JAX in float32 (faster on GPU, less accurate)")
    parser.add_argument("--platform", default=None, help="force a JAX platform, e.g. cpu or gpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    common.configure_jax(x64=not args.no_x64, platform=args.platform)

    spec = common.load_json(args.model)
    protocol = common.load_protocol(args.protocol)
    cell = common.build_model(spec, base_dir=Path(args.model).resolve().parent)

    simulate = common.jax.jit(common.make_simulator(cell, protocol))
    started = time.perf_counter()
    traces = np.asarray(simulate([], common.jnp.asarray(protocol.currents)))
    elapsed = time.perf_counter() - started

    if not np.all(np.isfinite(traces)):
        print("error: the simulation produced non-finite voltages; reduce delta_t or check parameters", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        traces=traces,
        time=protocol.time,
        currents=protocol.currents,
        labels=np.asarray(protocol.labels),
    )

    counts = common.hard_spike_count(traces, args.spike_threshold)
    print(f"model {spec.get('name', Path(args.model).stem)}: {len(cell.nodes)} compartments, groups {cell.group_names}")
    print(f"simulated {protocol.n_trials} trial(s) x {len(protocol.recordings)} recording(s) x {protocol.n_time} steps in {elapsed:.2f}s (includes compile)")
    for index, label in enumerate(protocol.labels):
        trial = traces[index]
        print(
            f"  {label}: v in [{trial.min():.1f}, {trial.max():.1f}] mV, "
            f"spikes per recording {counts[index].tolist()}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
