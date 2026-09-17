# The Jaxley training pipeline in detail

This document expands the stages in `SKILL.md`. Everything measured here was run with Jaxley
0.14.0, jax 0.11.1, optax 0.2.8 on CPU with float64; sections marked *illustrative* describe
workflows documented upstream (the Jaxley paper and tutorials) that were not executed for this
skill.

## 1. Data preparation

Jaxley's units are fixed: voltage **mV**, time **ms**, injected current **nA**, conductances
**S/cm^2**, lengths and radii **um**, specific capacitance **uF/cm^2**, axial resistivity
**ohm cm**. A recording only becomes a target after it is in those units, on the simulator's grid.

| Step | What to do | Why |
| --- | --- | --- |
| Time grid | Resample every sweep to the protocol's `delta_t` (0.025 ms default; 0.01 ms for fast axonal spikes). `np.interp` onto `np.arange(n_time) * delta_t` is enough for 10-50 kHz patch data. | `jx.integrate` returns exactly `len(jx.step_current(0, 0, 0, delta_t, t_max))` samples; the loss compares sample by sample. |
| Alignment | Put stimulus onset at the same index in every sweep and in the protocol (`i_delay`). Trim pre-stimulus baseline to 5-10 ms. | A 0.1 ms misalignment of a spike costs more MSE than a wrong conductance. |
| Junction potential | Subtract the liquid-junction potential before fitting (typically 10-15 mV for K-gluconate internals). | Reversal potentials are otherwise fitted to a biased voltage. |
| Holding current | Model a holding current as `i_offset` in the step definition, or fit `Leak_eLeak` to the resting potential first. | Otherwise the resting potential absorbs the error. |
| Sweeps to trials | One `steps` entry (or one `currents_file` row) per sweep; average repeated identical sweeps only if you also fit their variance elsewhere. | Trials are the batch dimension the pipeline vmaps over. |
| Noise | Low-pass filter lightly, or prefer the summary-statistics loss; never smooth so much that spike shapes change. | Raw MSE fits noise; summary statistics do not. |
| Held-out split | Reserve at least one amplitude and one duration the fit never sees, plus a noise or ramp waveform if available. | The only honest test of a fitted model (see stage 6). |

Save the result with `np.savez_compressed(path, traces=..., time=..., currents=..., labels=...)`,
`traces` shaped `(n_trials, n_recordings, n_time)`. `simulate_protocol.py` writes the same layout.

## 2. Model design decisions

- **Start simple.** Fit a single compartment before a ball-and-stick before an SWC morphology.
  Each stage tells you which parameters the data constrain; every parameter added without new
  data is a degeneracy.
- **Compartmentalisation.** Read the SWC with `ncomp=1` and apply the d_lambda rule
  (`_common.apply_d_lambda(cell, frequency=100.0, d_lambda=0.1)`): the number of compartments per
  branch is the smallest odd number such that each is shorter than 0.1 AC length constants at
  100 Hz. Fewer compartments are faster but distort dendritic attenuation; check that halving
  `d_lambda` does not change the fit. Each compartment's `length` is the branch length divided by
  its `ncomp`; setting `length` on a multi-compartment branch sets the *per-compartment* length.
- **Channel sets.** Built-in: `HH`, and the Pospischil et al. (2008) set `Na`, `K`, `Leak`,
  `Km`, `CaL`, `CaT` (plus the point-neuron `Izhikevich`, `AdEx`, `Fire`, `Rate`); calcium
  handling in `jaxley.pumps` (`CaNernstReversal`, `CaPump`, `CaFaradayConcentrationChange`).
  Published sets in `jaxley-mech`:
  `jaxley_mech.channels.l5pc` (Hay et al. layer-5 pyramidal cell: `NaTs2T`, `NaTaT`, `NapEt2`,
  `SKv3_1`, `SKE2`, `KPst`, `KTst`, `M`, `H`, `CaHVA`, `CaLVA`, `CaPump`, `CaNernstReversal`),
  `hodgkin52`, `fohlmeister97` (retinal ganglion), `kamiyama09`, `aoyama00`, `usui96`,
  `torre90`, `liu04`, `benison01`, `chen24`. In a spec, reference them by dotted path,
  e.g. `"channel": "jaxley_mech.channels.l5pc.NaTs2T"`. Custom channels subclass
  `jaxley.channels.Channel` (tutorial 05).
- **What to fit.** Maximal conductances first; passive properties (`radius`, `axial_resistivity`,
  `capacitance`, `Leak_gLeak`, `Leak_eLeak`) from sub-threshold sweeps; reversal potentials only
  when the ionic concentrations are unknown; kinetics (`vt` shifts, time constants) last and only
  with data that constrain them. Keep `v_init` at the measured resting potential and let
  `init_states()` compute steady-state gating - a fit that has to spend its first 20 ms relaxing
  is fitting the wrong thing.
- **Temperature.** Channel models carry their own temperature or q10 handling; the L5PC example
  sets `CaNernstReversal().channel_constants["T"] = 307.15`. Match the recording temperature.

## 3. Protocol design

Fit on several amplitudes spanning sub-threshold, threshold, and strong drive: sub-threshold
sweeps constrain passive properties, near-threshold sweeps constrain the sodium/leak balance, and
strong sweeps constrain potassium and adaptation currents. Include at least one long step
(hundreds of ms) if the cell adapts. Record from every site you have data for - the bundled
ball-and-stick model cannot separate dendritic leak from dendrite radius with a somatic recording
alone, and the same is true of any dendritic parameter.

Trials are vmapped, so 3 or 30 trials compile to one program; simulation time scales roughly
linearly with `n_trials * n_time` on CPU and much better than linearly on GPU.

## 4. Parameterisation and bounds

Optimise in the transformed space of `jx.ParamTransform` and map back inside the loss. Bounds
that worked for HH-type cells (upstream L5PC example and this skill's tests):

| Parameter | Typical bounds (S/cm^2 unless noted) | Transform |
| --- | --- | --- |
| Somatic transient Na (`HH_gNa`, `NaTs2T_gNaTs2T`) | 0.01 - 0.5 (soma), up to 4.0 in the axon initial segment | `SigmoidTransform` |
| Delayed-rectifier K (`HH_gK`, `SKv3_1_gSKv3_1`) | 0.001 - 0.2 (soma), up to 2.0 (axon) | `SigmoidTransform` |
| Leak (`Leak_gLeak`, `HH_gLeak`) | 1e-5 - 1e-3 | `SigmoidTransform` |
| Ca channels (`CaHVA_gCaHVA`, `CaLVA_gCaLVA`) | 0 - 1e-3 / 0 - 1e-2 | `SigmoidTransform` |
| Ca pump (`CaPump_gamma`, `CaPump_decay`) | 5e-4 - 0.05, 20 - 1000 ms | `SigmoidTransform` |
| `radius` (um) | 0.3 - 3 dendrite, 5 - 15 soma | `SigmoidTransform` |
| `axial_resistivity` (ohm cm) | 50 - 300 | `SigmoidTransform` |
| Reversal potentials (mV) | +/- 15 around the Nernst value | `AffineTransform` or identity |

`SoftplusTransform(lower)` for one-sided positivity, `IdentityTransform` for genuinely free
parameters, `ChainTransform` and `AffineTransform` to rescale. Log-spaced bounds are not built in;
approximate them with `CustomTransform(lambda x: lower * (upper/lower) ** sigmoid(x), inverse)`
(*illustrative*).

Parameter sharing: `view.make_trainable(key)` shares one value across the view;
`view.branch("all").comp("all").make_trainable(key)` gives each compartment its own; groups of
different sizes can share (changelog 0.6.0). Distance-dependent conductances (the L5PC H-current)
are set with `jaxley.morphology.distance_direct(cell.soma.branch(0).comp(0), cell)` and can be
made trainable through a low-dimensional parametrisation with `data_set` inside the loss.

## 5. Loss design

| Loss (`--loss`) | Formula | Use when | Watch out |
| --- | --- | --- | --- |
| `mse` | mean over trials, recordings, samples of (v - v_target)^2 after `--loss-start-ms` | Clean, aligned traces; final refinement | Rugged: spike timing dominates; needs many starts |
| `summary` | mean absolute error of per-window mean and std, divided by (2 mV, 1 mV) | Noisy or jittery data, coarse fit, many parameters | Blind to spike timing within a window; refine with `mse` |
| `spike_count` | squared error of the soft spike count (total positive variation of `sigmoid((v - 0)/2 mV)`) | Matching f-I curves and firing regimes | Nearly zero gradient away from threshold events; finishes at count agreement with the wrong shape (12 mV RMSE in our run) |

Combine them as a weighted sum when one alone is not enough (*illustrative*):

```python
def loss_fn(opt_params):
    v = batched(transform.forward(opt_params), currents)
    return (common.summary_loss(v, target)
            + 0.01 * common.mse_loss(v, target)
            + 0.1 * common.spike_count_loss(v, target))
```

Standardise every term to O(1) before adding them; a loss-scaled learning rate (Polyak) assumes
the loss is O(1) at a bad fit and near 0 at a good one. For sub-threshold sweeps, MSE on the raw
trace is ideal. For calcium targets, see section 9.

## 6. Optimisation

Measured on the bundled `hh_soma` problem (3 parameters, 3 trials, 2401 samples):

| Setting | Effect |
| --- | --- |
| Screening 32 random starts with `jax.vmap(loss_fn)` | 0.5 s; picks starts with loss 300-430 mV^2 instead of 700+ |
| Descending from the single best start (150 Adam steps, lr 0.05) | ~5 s; converged in 3 of 4 seeds, 16 mV RMSE local minimum in the fourth |
| Descending from the best 4 in parallel (`--top-k 4`) | 6.6 s; converged in 6 of 6 seeds, RMSE 0.02-0.7 mV |
| Adam lr 0.1 from a 50%-perturbed start | 40 steps to loss < 0.2 mV^2, parameters within 3% |

Recommendations:

- **Adam** (`optax.adam`, learning rate 0.01-0.1 in transformed space, `clip_by_global_norm(1.0)`)
  for MSE and spike losses. Larger steps than 0.1 overshoot the sigmoid into saturation.
- **Polyak-style normalised SGD** (`_common.polyak_step`: gradient divided by its norm to the
  power `beta` ~ 0.9, learning rate `mu * loss`, `mu = 0.1 * sqrt(n_params)` upstream) for
  standardised summary-statistic losses on morphologically detailed cells, where raw gradients
  vary over orders of magnitude between steps. Always guard the norm with an epsilon.
- **Random starts, then parallel descent.** `--n-starts 32 --top-k 4` is a sensible default for
  <= 10 parameters; `--n-starts 256 --top-k 16` on GPU for 20+ parameters. Compare the winners:
  agreeing losses with disagreeing parameters mean the protocol does not identify them.
- **Two-stage fits.** `--loss summary` from random starts, then `--loss mse --init spec` on the
  resulting `fitted_model.json`, is more robust than MSE alone when the traces are noisy.
- **Schedules** (*illustrative*): `optax.cosine_decay_schedule(0.05, 300)` or
  `optax.exponential_decay(0.05, 100, 0.5)` passed to `optax.adam` help the last few percent.
- **Stopping.** `--patience 40` stops a stalled fit; the trainer always keeps the best parameters
  seen, not the last.

## 7. Memory, speed, and hardware

- Backward-pass memory grows with `n_trials * n_compartments * n_time`. Two checkpoint levels
  (`checkpoint_lengths = [ceil(sqrt(n_time))] * 2`, `--checkpoint-levels 2`) bring it to
  O(sqrt(n_time)) for roughly 30% more compute; three levels for very long traces.
- Compile once, reuse: keep `loss_fn` and the vmapped simulator alive across steps; changing the
  protocol length, the number of trials, or `checkpoint_lengths` triggers recompilation.
- On CPU the first call of the L5PC-scale model compiles in seconds to a minute; `bwd_euler` and
  `crank_nicolson` are the fastest CPU solvers.
- On GPU set `XLA_PYTHON_CLIENT_MEM_FRACTION=.8`, batch many parameter sets or trials, and consider
  `solver="exp_euler"` with `cell.customize_solver_exp_euler(exp_euler_transition=cell.build_exp_euler_transition_matrix(delta_t))`
  when passive properties are not trained (up to 5x faster than `bwd_euler` upstream).
  `exp_euler` is slow on CPU and not optimised for networks.
- Float32 (`--no-x64`) roughly doubles GPU throughput; re-check the fit in float64 before
  reporting it.

## 8. Networks and task training

```python
net = jx.Network([cell for _ in range(n_cells)])
jx.sparse_connect(net.cell(range(n_in)), net.cell(range(n_in, n_cells)), IonotropicSynapse(), p=0.2)
net.IonotropicSynapse.edge("all").make_trainable("IonotropicSynapse_gS")
net.cell("all").branch("all").comp("all").make_trainable("HH_gNa")
```

- `jx.fully_connect`, `jx.sparse_connect(p=...)`, and `jx.connectivity_matrix_connect(matrix)`
  add synapses between cell views; `net.<SynapseName>.edge("all")` indexes them, and `net.edges`
  lists them. Synaptic conductances default to 1e-4 S/cm^2 - scale to the postsynaptic area.
- Inputs: one `data_stimulate` call per input cell inside `simulate(params, inputs)`; the upstream
  tutorial maps a 2-D dataset row to two step-current amplitudes with
  `jx.datapoint_to_step_currents` and trains a mean-absolute-error classifier on the mean output
  voltage (*illustrative* here; the tutorial runs it).
- Readouts: the recorded output voltage, its mean over a window, or the soft spike count.
  `TanhRateSynapse` gives rate-based networks when spikes are not needed.
- Record only what the loss needs (`net.delete_recordings()` first); every recording costs
  memory in the backward pass.
- `jx.Network(cells, vectorize_cells=...)` controls whether identical cells are solved in one
  vectorised block; keep the default unless profiling says otherwise.

## 9. Calcium-imaging targets (*illustrative*)

The Jaxley paper fits models to two-photon calcium recordings. The pieces exist in the library:
insert `CaL`/`CaT` (or the `jaxley-mech` L5PC `CaHVA`/`CaLVA`), `jaxley.pumps.CaFaradayConcentrationChange`
and `jaxley.pumps.CaNernstReversal`, optionally `cell.diffuse("CaCon_i")`, then
`record("CaCon_i")` and convolve the recorded concentration with an indicator kernel
(`jnp.convolve` with a double exponential, tau_rise ~ 50 ms, tau_decay ~ 400 ms for GCaMP6s) before
comparing dF/F. Fit the kernel amplitude and baseline jointly with the biophysics. This skill's
scripts record any state name (`"state": "CaCon_i"` in a protocol recording) but ship no indicator
model.

## 10. Combining with other inference methods

`jax.vmap` over parameter sets makes Jaxley a fast simulator for random search, genetic
algorithms, and simulation-based inference (SBI): draw parameters, simulate the batch, train the
posterior, then use gradient descent from posterior samples for the final point estimate. The
upstream L5PC example recommends exactly this hybrid for models with few parameters.

## 11. Reproducibility checklist

- `jaxley.__version__`, `jax.__version__`, `optax.__version__`, platform (CPU/GPU) and float64 flag.
- The model spec, the protocol, the target file hash, and the held-out protocol.
- Bounds and transforms for every trainable; seed; `--n-starts`, `--top-k`, optimizer, learning
  rate, steps, checkpoint levels (all stored in `fit_result.json` under `config`).
- Training loss curve, best step, winner index, the other candidates' final losses.
- Held-out RMSE, spike-count agreement, first-spike latency error (from `evaluate_fit.py`).
- A convergence check: the fit does not move when `delta_t` is halved or `d_lambda` is reduced.
