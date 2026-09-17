# Troubleshooting Jaxley fits

Symptoms observed while building and testing this skill (Jaxley 0.14.0, jax 0.11.1, optax 0.2.8,
CPU, float64), with the cause that was actually found and the fix that worked.

## The loss becomes NaN

| Cause | How to recognise it | Fix |
| --- | --- | --- |
| Gradient normalised by a vanishing norm | Parameters pinned to a bound (transformed values with magnitude > 8), `l2_norm(grads)` ~ 1e-40, then NaN on the next step | Divide by `l2_norm(grads) + eps` (`_common.polyak_step` does), clip updates, and widen or re-centre the bounds |
| Loss-scaled learning rate on an unstandardised loss | First Polyak step of size ~100 in transformed space when the loss is O(100) mV^2 | Use Polyak only with standardised losses (`summary`); use Adam with `clip_by_global_norm(1.0)` for `mse` |
| Forward Euler on a multicompartment cell | Voltages explode within a few ms | `solver="bwd_euler"` or `"crank_nicolson"` |
| Time step too coarse for fast kinetics | Fine at `delta_t=0.025`, NaN or ringing at 0.1 | Reduce `delta_t`; check the fit is unchanged at half the step |
| float32 with stiff channels | NaN only after `--no-x64` | Keep float64 for fitting; use float32 only for GPU screening |

The simulator itself stayed finite in every test here, including with every conductance at its
bound - suspect the optimizer before the model.

## The fit converges to a bad minimum

- **Single start on a spiking-trace MSE.** With one start the bundled 3-parameter HH problem ended
  at 16 mV RMSE in about one seed in four. Screen many random draws (`--n-starts 32` costs 0.5 s)
  and descend from several (`--top-k 4`); the winner was frequently not the best-screened start.
- **Spike-count loss looks converged but the traces are wrong.** The soft count is piecewise
  nearly flat; it matched the counts with 12 mV RMSE. Use it to enter the right regime, then
  refine with `--loss mse --init spec` on `fitted_model.json`.
- **Summary loss plateaus above zero.** Mean/std per window cannot resolve spike timing. Refine with
  `mse`, or use more windows (`--summary-windows 8`).
- **Parameters wander while the loss stays flat** (compare winners across seeds): the protocol does
  not constrain them. Add sweeps (sub-threshold for passive, long steps for adaptation), record from
  more sites, or fix the unidentified parameter to a literature value.
- **Learning rate too high**: loss oscillates and transformed parameters jump between -8 and 8.
  Halve `--learning-rate`; 0.05 is a good Adam default in sigmoid space.

## `warning: [...] sit at a bound`

The transformed value exceeds 8 in magnitude, i.e. the parameter is within 0.03% of its bound and
its gradient is ~0. Either the truth lies outside the bound (widen it) or the model compensates for
a missing mechanism with an extreme value (revisit the channel set). Fits with saturated
parameters generalise poorly.

## Shape and value errors

| Message | Cause | Fix |
| --- | --- | --- |
| `target traces have shape (a, b, c), protocol expects (n_trials, n_recordings, n_time)` | Target saved on a different grid or with a different number of sweeps/sites | Resample to the protocol's `delta_t`/`t_max`; one recording entry per recorded site |
| `currents_file has N samples per trial, but t_max=... needs M` | Waveform length != `round(t_max/delta_t) + 1` | Regenerate the waveform on the protocol grid. Jaxley itself would not complain: it zero-pads a short stimulus and truncates a long one silently |
| `group 'axon' does not exist; available groups: [...]` | SWC has no type-2 points, or `add_to_group` was never called | Use an existing group or create one |
| `unknown channel 'Nav'` | Not a `jaxley.channels` class name | Use the listed built-ins or a dotted `jaxley_mech...` path |
| `ValueError: Must have equal len keys and value` from `write_trainables` | Two trainables share a key | Export via the spec (`_common.spec_with_parameters`) or `set` per view |
| `AttributeError: 'NoneType' object has no attribute 'keys'` from `dir(cell)` | Upstream 0.14.0 quirk in `Module.__dir__` | Use `cell.group_names`, `cell.nodes.columns`, `cell.show()` |
| `random initialisation needs both lower and upper bounds` | `--n-starts` with a one-sided or unbounded trainable | Add bounds, or start from the spec values without screening |

## Speed and memory

- **Every step recompiles**: something in the traced arguments changes shape - a Python `int` that
  varies, a different number of trials, or `checkpoint_lengths` computed from a changing length.
  Keep the protocol fixed inside one `loss_fn`.
- **Out of memory in the backward pass**: enable checkpointing (`--checkpoint-levels 2`), reduce
  `n_trials` per step (mini-batch trials), record fewer sites, or shorten `t_max` and fit windows
  separately.
- **Slow on GPU for networks**: 0.11.2 fixed sequential processing of cells; upgrade past it. For
  single cells with fixed passive properties, `exp_euler` with a precomputed transition matrix is
  up to 5x faster than `bwd_euler` upstream; on CPU it is slower.
- **Slow on CPU**: `bwd_euler`/`crank_nicolson` are fastest; batch trials with `vmap` rather than
  looping; screen starts with `vmap` instead of sequential runs (32 starts took the time of one).

## Numerical checks before trusting a fit

1. Halve `delta_t`; the loss and parameters should move by less than the seed-to-seed spread.
2. Reduce `d_lambda` (more compartments) for SWC models; same criterion.
3. Re-simulate `fitted_model.json` with `simulate_protocol.py` and confirm it reproduces
   `fitted_traces.npz` (guards against a spec export bug).
4. Evaluate on the held-out protocol; compare spike counts and first-spike latencies, not just
   RMSE - a near-threshold held-out step is the most sensitive test.
5. Repeat the fit with two more seeds; report the spread of the winning parameters.
