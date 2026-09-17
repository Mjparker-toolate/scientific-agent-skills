#!/usr/bin/env python3
"""Fit a Jaxley model's trainable parameters to target traces with gradient descent.

The pipeline, end to end: build the cell from a JSON spec, register the spec's
``trainable`` entries and their bounded transforms, simulate every trial of the
protocol in one ``vmap``-ed and ``jit``-compiled call, compare with the target
traces through a differentiable loss, and update the parameters in
unconstrained (transformed) space with Adam or the Polyak-style normalised SGD
from the Jaxley L5PC example.

Initialisation matters more than the optimizer on spiking traces, whose voltage
loss is rugged. ``--n-starts N`` screens N uniform draws inside the bounds in a
single vectorised call, and ``--top-k K`` then runs gradient descent from the K
best draws *in parallel* (one more ``vmap``), keeping whichever ends lowest.
A candidate whose loss turns non-finite is frozen rather than poisoning the rest.

Outputs in ``--out``: ``fit_result.json`` (fitted values, loss history, config),
``fitted_model.json`` (the spec with fitted values applied, ready for
``simulate_protocol.py`` / ``evaluate_fit.py``), and ``fitted_traces.npz``.

Example::

    python scripts/fit_biophysics.py --model assets/hh_soma.json \
        --protocol assets/step_protocol.json --target target.npz \
        --out fit/ --init random --n-starts 32 --top-k 4 --steps 150
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
    parser.add_argument("--model", required=True, help="model specification JSON with a `trainable` block")
    parser.add_argument("--protocol", required=True, help="protocol JSON")
    parser.add_argument("--target", required=True, help=".npz with `traces` of shape (n_trials, n_recordings, n_time)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--loss", choices=sorted(common.LOSSES), default="mse", help="voltage MSE, windowed summary statistics, or soft spike counts (default mse)")
    parser.add_argument("--loss-start-ms", type=float, default=0.0, help="ignore the first N ms in the MSE loss (default 0)")
    parser.add_argument("--summary-windows", type=int, default=4, help="windows for --loss summary (default 4)")
    parser.add_argument("--spike-threshold", type=float, default=common.SPIKE_THRESHOLD_MV, help="mV threshold for spike losses/metrics")
    parser.add_argument("--spike-width", type=float, default=common.SOFT_SPIKE_WIDTH_MV, help="mV softness of the differentiable spike count")
    parser.add_argument("--optimizer", choices=("adam", "polyak"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Adam step size in transformed space, or Polyak mu (default 0.05)")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Adam global gradient-norm clip; 0 disables (default 1.0)")
    parser.add_argument("--beta", type=float, default=0.9, help="Polyak gradient-normalisation exponent (default 0.9)")
    parser.add_argument("--steps", type=int, default=200, help="gradient steps (default 200)")
    parser.add_argument("--patience", type=int, default=0, help="stop after N steps without improvement of the best candidate; 0 disables")
    parser.add_argument("--init", choices=("spec", "random"), default="spec", help="candidates are the spec values plus --n-starts random draws (spec), or random draws only")
    parser.add_argument("--n-starts", type=int, default=0, help="uniform random draws inside the bounds to screen (default 0)")
    parser.add_argument("--top-k", type=int, default=4, help="descend in parallel from the K best screened candidates (default 4)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-levels", type=int, default=0, help="gradient checkpointing levels (0 = off, 2 = O(sqrt T) memory)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--no-x64", action="store_true", help="run JAX in float32")
    parser.add_argument("--platform", default=None, help="force a JAX platform, e.g. cpu or gpu")
    return parser.parse_args(argv)


def build_loss(args: argparse.Namespace, protocol: common.Protocol, target):
    start_index = int(round(args.loss_start_ms / protocol.delta_t))
    if args.loss == "mse":
        return lambda pred: common.mse_loss(pred, target, start_index=start_index)
    if args.loss == "summary":
        return lambda pred: common.summary_loss(pred, target, n_windows=args.summary_windows)
    return lambda pred: common.spike_count_loss(pred, target, threshold=args.spike_threshold, width=args.spike_width)


def select(pytree, index):
    return common.jax.tree_util.tree_map(lambda leaf: leaf[index], pytree)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.init == "random" and args.n_starts < 1:
        print("error: --init random needs --n-starts >= 1", file=sys.stderr)
        return 2
    common.configure_jax(x64=not args.no_x64, platform=args.platform)
    jax, jnp, optax = common.jax, common.jnp, common.optax

    spec = common.load_json(args.model)
    protocol = common.load_protocol(args.protocol)
    target_np = common.load_array(args.target, key="traces")
    expected = (protocol.n_trials, len(protocol.recordings), protocol.n_time)
    if tuple(target_np.shape) != expected:
        print(f"error: target traces have shape {target_np.shape}, protocol expects {expected}", file=sys.stderr)
        return 2
    target = jnp.asarray(target_np)

    cell = common.build_model(spec, base_dir=Path(args.model).resolve().parent)
    transform = common.add_trainables(cell, spec)
    params = cell.get_parameters()
    n_params = int(sum(np.size(value) for group in params for value in group.values()))
    simulate = common.make_simulator(cell, protocol, checkpoint_levels=args.checkpoint_levels)
    currents = jnp.asarray(protocol.currents)
    loss_of_prediction = build_loss(args, protocol, target)

    def loss_fn(opt_params):
        return loss_of_prediction(simulate(transform.forward(opt_params), currents))

    print(f"fitting {n_params} parameter(s) in {len(params)} group(s) of {spec.get('name', Path(args.model).stem)} "
          f"on {protocol.n_trials} trial(s), loss={args.loss}, optimizer={args.optimizer}")

    # --- candidates: spec values and/or uniform draws inside the bounds, screened once
    spec_start = transform.inverse(params)
    if args.n_starts > 0:
        candidates = transform.inverse(common.sample_within_bounds(spec, params, args.n_starts, args.seed))
        if args.init == "spec":
            candidates = jax.tree_util.tree_map(
                lambda drawn, current: jnp.concatenate([current[None], drawn], axis=0), candidates, spec_start
            )
    else:
        candidates = jax.tree_util.tree_map(lambda leaf: leaf[None], spec_start)
    started = time.perf_counter()
    screen = np.asarray(jax.jit(jax.vmap(loss_fn))(candidates))
    screen = np.where(np.isfinite(screen), screen, np.inf)
    if not np.isfinite(screen).any():
        print("error: every candidate start produced a non-finite loss; check delta_t, bounds and v_init", file=sys.stderr)
        return 2
    n_keep = max(1, min(args.top_k, int(np.isfinite(screen).sum())))
    order = np.argsort(screen)[:n_keep]
    batch = jax.tree_util.tree_map(lambda leaf: leaf[order], candidates)
    print(f"screened {screen.size} candidate(s) in {time.perf_counter() - started:.1f}s; "
          f"descending from the best {n_keep}: initial losses {np.round(screen[order], 4).tolist()}")

    # --- one optimizer step for one candidate; vmapped over the kept candidates
    if args.optimizer == "adam":
        optimizer = common.make_adam(args.learning_rate, args.clip_norm)
    else:
        optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=args.learning_rate)

    def train_step(opt_params, opt_state, best_loss, best_params):
        loss_value, grads = jax.value_and_grad(loss_fn)(opt_params)
        grad_norm = common.l2_norm(grads)
        ok = jnp.isfinite(loss_value) & jnp.isfinite(grad_norm)
        improved = ok & (loss_value < best_loss)
        best_loss = jnp.where(improved, loss_value, best_loss)
        best_params = jax.tree_util.tree_map(lambda best, current: jnp.where(improved, current, best), best_params, opt_params)
        if args.optimizer == "polyak":
            grads, learning_rate = common.polyak_step(grads, loss_value, mu=args.learning_rate, beta=args.beta)
            opt_state.hyperparams["learning_rate"] = learning_rate
        updates, new_state = optimizer.update(grads, opt_state, opt_params)
        new_params = optax.apply_updates(opt_params, updates)
        # A diverged candidate keeps its last finite parameters instead of spreading NaN.
        new_params = jax.tree_util.tree_map(lambda new, old: jnp.where(ok, new, old), new_params, opt_params)
        new_state = jax.tree_util.tree_map(lambda new, old: jnp.where(ok, new, old), new_state, opt_state)
        return new_params, new_state, best_loss, best_params, loss_value, grad_norm

    batched_step = jax.jit(jax.vmap(train_step))
    batched_loss = jax.jit(jax.vmap(loss_fn))

    opt_params = batch
    opt_state = jax.vmap(optimizer.init)(batch)
    best_loss = jnp.full((n_keep,), jnp.inf)
    best_params = batch
    history: list[list[float]] = []
    grad_norms: list[list[float]] = []
    stalled, stop_reason = 0, "completed all steps"
    started = time.perf_counter()
    for step in range(args.steps):
        previous_best = float(jnp.min(best_loss))
        opt_params, opt_state, best_loss, best_params, losses, norms = batched_step(opt_params, opt_state, best_loss, best_params)
        losses_np, norms_np = np.asarray(losses), np.asarray(norms)
        history.append(losses_np.tolist())
        grad_norms.append(norms_np.tolist())
        if not np.isfinite(losses_np).any():
            stop_reason = f"every candidate diverged at step {step}; reverting to the best parameters seen"
            break
        stalled = 0 if float(jnp.min(best_loss)) < previous_best - 1e-12 else stalled + 1
        if args.patience and stalled >= args.patience:
            stop_reason = f"no improvement for {args.patience} steps"
            break
        if step % args.log_every == 0 or step == args.steps - 1:
            leader = int(np.nanargmin(np.where(np.isfinite(losses_np), losses_np, np.inf)))
            print(f"  step {step:4d}  loss {losses_np[leader]:.6g} (candidate {leader}; "
                  f"all finite: {np.isfinite(losses_np).all()})  |grad| {norms_np[leader]:.3g}")
    elapsed = time.perf_counter() - started

    # The last update was never scored, so evaluate the final parameters once more.
    final_losses = np.asarray(batched_loss(opt_params))
    final_losses = np.where(np.isfinite(final_losses), final_losses, np.inf)
    best_loss_np = np.asarray(best_loss)
    take_final = final_losses < best_loss_np
    best_loss_np = np.where(take_final, final_losses, best_loss_np)
    best_params = jax.tree_util.tree_map(
        lambda final, best: jnp.where(jnp.asarray(take_final).reshape((-1,) + (1,) * (final.ndim - 1)), final, best),
        opt_params,
        best_params,
    )
    winner = int(np.argmin(best_loss_np))
    best_single = select(best_params, winner)
    if not np.isfinite(best_loss_np[winner]):
        print("error: no candidate ever produced a finite loss", file=sys.stderr)
        return 1

    pinned = common.saturation_report(best_single)
    fitted = transform.forward(best_single)
    records = common.parameters_to_records(spec, fitted)
    fitted_traces = np.asarray(jax.jit(simulate)(fitted, currents))
    rmse = float(np.sqrt(np.mean((fitted_traces - target_np) ** 2)))
    winner_history = [row[winner] for row in history]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fitted_spec = common.spec_with_parameters(spec, records)
    fitted_spec["name"] = f"{spec.get('name', Path(args.model).stem)}_fitted"
    common.dump_json(fitted_spec, out / "fitted_model.json")
    common.dump_json(
        {
            "model": str(Path(args.model).resolve()),
            "protocol": str(Path(args.protocol).resolve()),
            "target": str(Path(args.target).resolve()),
            "config": vars(args),
            "n_parameters": n_params,
            "n_candidates_screened": int(screen.size),
            "n_candidates_descended": n_keep,
            "screened_losses": screen.tolist(),
            "steps_run": len(history),
            "stop_reason": stop_reason,
            "seconds": elapsed,
            "winner": winner,
            "initial_loss": float(screen[order][winner]),
            "best_loss": float(best_loss_np[winner]),
            "candidate_best_losses": best_loss_np.tolist(),
            "voltage_rmse_mV": rmse,
            "saturated_parameters": pinned,
            "fitted_parameters": records,
            "loss_history": winner_history,
            "grad_norm_history": [row[winner] for row in grad_norms],
            "loss_history_all_candidates": history,
        },
        out / "fit_result.json",
    )
    np.savez_compressed(
        out / "fitted_traces.npz",
        traces=fitted_traces,
        time=protocol.time,
        currents=protocol.currents,
        labels=np.asarray(protocol.labels),
    )

    print(f"{stop_reason}; {len(history)} step(s) x {n_keep} candidate(s) in {elapsed:.1f}s")
    print(f"candidate {winner} won: loss {screen[order][winner]:.6g} -> {best_loss_np[winner]:.6g}; "
          f"voltage RMSE {rmse:.3f} mV; other candidates ended at {np.round(np.delete(best_loss_np, winner), 4).tolist()}")
    for record in records:
        value = record["value"]
        shown = f"{value:.6g}" if isinstance(value, float) else f"{len(value)} values, mean {np.mean(value):.6g}"
        print(f"  {record['group']}.{record['key']} = {shown}")
    if pinned:
        print(f"warning: {pinned} sit at a bound (|transformed value| > 8); widen or re-centre their bounds")
    print(f"wrote {out / 'fit_result.json'}, {out / 'fitted_model.json'}, {out / 'fitted_traces.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
