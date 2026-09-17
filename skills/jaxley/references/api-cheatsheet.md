# Jaxley 0.14 API cheat-sheet

Signatures and defaults below were read from the installed Jaxley 0.14.0 (`inspect.signature`)
and exercised by this skill's scripts and tests. Upstream API reference:
https://jaxley.readthedocs.io/en/latest/jaxley.html.

## Module hierarchy

```python
import jaxley as jx
comp   = jx.Compartment()
branch = jx.Branch(comp, ncomp=4)                    # or a list of Compartments
cell   = jx.Cell([branch_a, branch_b], parents=[-1, 0])   # parents[i]: parent branch of branch i, -1 = root
net    = jx.Network([cell_1, cell_2], vectorize_cells=None)
cell   = jx.read_swc(fname, ncomp, max_branch_len=None, min_radius=None, assign_groups=True,
                     backend="graph", ignore_swc_tracing_interruptions=True, relevant_type_ids=None)
```

`read_swc` assigns groups `soma` (type 1), `axon` (2), `basal` (3), `apical` (4) and ignores
type ids > 4 by default. Every module exposes `.nodes` (a pandas DataFrame with one row per
compartment: `length`, `radius`, `axial_resistivity`, `capacitance`, `v`, `x/y/z`, channel flags,
channel parameters and states, `global_*_index` columns) and `.edges` (one row per synapse).

## Views and indexing

| Expression | Selects |
| --- | --- |
| `cell.branch(0)`, `cell.branch([0, 2])`, `cell.branch("all")` | branches |
| `view.comp(i)` / `view.loc(0.5)` | compartment by index / by normalised position along the branch |
| `net.cell(i)` | cells of a network |
| `net.SynapseName.edge("all")`, `net.edge(i)` | synapses |
| `cell.soma`, `cell.apical`, `cell.dendrite` | groups (`cell.group_names` lists them; `view.add_to_group("name")` creates one) |
| `net.exc.soma` | intersections of groups (0.8.2+) |
| `cell.select(nodes=...)`, `cell.scope("local" or "global")` | arbitrary node subsets, index scope |

`cell.branches` iterates branch views (used by the d_lambda rule). `dir(module)` raises an
`AttributeError` in 0.14.0 when no groups exist (`group_nodes` is `None`); use `cell.group_names`.

## Module methods used in training

| Method | Purpose |
| --- | --- |
| `insert(channel_or_pump)`, `delete(channel)` | add / remove a mechanism on the view |
| `set(key, value)` | set a parameter or state; `value` may be an array with one entry per compartment of the view |
| `data_set(key, value, param_state)` | traced version of `set` for use inside `jit`/`vmap`; pass the returned `param_state` to `jx.integrate(param_state=...)` |
| `make_trainable(key, init_val=None, verbose=True)` | register a shared parameter for the view; call on `view.branch("all").comp("all")` for one per compartment |
| `get_parameters()` | list of `{key: array}` in `make_trainable` order - the pytree the loss differentiates |
| `get_all_parameters(pstate)`, `get_all_states(pstate)` | full parameter / state dictionaries |
| `write_trainables(params)` | write fitted values back into `.nodes`; fails when two trainables share a key |
| `delete_trainables()` | reset |
| `stimulate(current, verbose=True)`, `delete_stimuli()` | fixed stimulus stored on the module |
| `data_stimulate(current, data_stimuli=None, verbose=False)` | traced stimulus; chain calls by passing the previous return value; hand the result to `jx.integrate(data_stimuli=...)` |
| `clamp(state, values)`, `data_clamp(state, values, data_clamps=None)` | voltage / state clamp (multi-state clamps fixed in 0.14.0) |
| `record(state="v", verbose=True)`, `delete_recordings()` | what `jx.integrate` returns, in call order |
| `init_states(delta_t=0.025)` | channel states at steady state for the current `v` |
| `set_ncomp(n, min_radius=None, initialize=True)` | recompartmentalise a branch; pass `initialize=False` in loops and call `cell.initialize()` once |
| `compute_xyz()`, `compute_compartment_centers()` | coordinates for plotting and distances |
| `vis(ax=None, color="k", dims=(0, 1), type="line")`, `show(param_names=None, ...)` | plot / tabulate |
| `copy(reset_index=False, as_module=False)` | duplicate a module or view |
| `diffuse(state)` | axial diffusion of an ion concentration (`cell.set("axial_diffusion_CaCon_i", ...)`) |
| `customize_solver_exp_euler(exp_euler_transition=None)`, `build_exp_euler_transition_matrix(delta_t, axial_conductances=None)` | precompute the exponential-Euler transition matrix (GPU) |

## Simulation

```python
jx.integrate(module, params=[], *, param_state=None, data_stimuli=None, data_clamps=None,
             t_max=None, delta_t=0.025, solver="bwd_euler", voltage_solver="jaxley.dhs",
             checkpoint_lengths=None, all_states=None, return_states=False) -> Array
```

- Returns `(n_recordings, n_time)`; under `jax.vmap` over trials `(n_trials, n_recordings, n_time)`.
  `n_time = len(jx.step_current(0, 0, 0, delta_t, t_max))` = `round(t_max / delta_t) + 1`.
- `solver`: `bwd_euler` (default, stable), `crank_nicolson` (stable, second order), `fwd_euler`
  (unstable for multicompartment cells), `exp_euler` (GPU, fixed passive properties).
- `checkpoint_lengths`: list of segment lengths per nesting level; `[ceil(sqrt(T))] * 2` for two
  levels. Product must cover `n_time`.
- `all_states` / `return_states`: continue a simulation from a saved state or return the final
  state; `jaxley.integrate.build_init_and_step_fn(module)` exposes the single-step function and
  `jaxley.utils.dynamics.build_dynamic_state_utils(module)` flattens the true ODE states (0.13.0).

Stimuli: `jx.step_current(i_delay, i_dur, i_amp, delta_t, t_max, i_offset=0.0)`;
`jx.datapoint_to_step_currents(i_delay, i_dur, i_amp_array, delta_t, t_max, i_offset=0.0)` builds
one current per amplitude for batched inputs.

## Channels (`jaxley.channels`) and their parameters

| Channel | Parameters (defaults) | States |
| --- | --- | --- |
| `HH` | `HH_gNa` 0.12, `HH_gK` 0.036, `HH_gLeak` 0.0003, `HH_eNa` 50, `HH_eK` -77, `HH_eLeak` -54.3 | `HH_m`, `HH_h`, `HH_n` |
| `Na` (Pospischil et al. 2008 set) | `Na_gNa` 0.05, `eNa` 50, `vt` -60 | gating |
| `K` (Pospischil) | `K_gK` 0.005, `eK` -90, `vt` -60 | gating |
| `Leak` (Pospischil) | `Leak_gLeak` 1e-4, `Leak_eLeak` -70 | - |
| `Km`, `CaL`, `CaT` (Pospischil) | slow K, high- and low-threshold Ca currents | gating |
| `Izhikevich`, `AdEx` (0.14.0), `Fire`, `Rate` (`jaxley.channels.non_capacitive`) | simplified / point-neuron dynamics | model-specific |
| `jaxley.pumps`: `CaNernstReversal`, `CaPump`, `CaFaradayConcentrationChange` | calcium concentration and reversal dynamics | `CaCon_i`, `CaCon_e`, ... |

Recorded currents use `channel.current_name` (`i_HH`, ...). `jaxley-mech` adds published channel
sets by module: `l5pc`, `hodgkin52`, `fohlmeister97`, `kamiyama09`, `aoyama00`, `usui96`,
`torre90`, `liu04`, `benison01`, `chen24`. Custom mechanisms subclass `Channel` and implement
`update_states`, `compute_current`, `init_state`; NMODL files can be converted with the how-to
guide "Import channels from NEURON".

## Synapses and connectivity

| Synapse (`jaxley.synapses`) | Parameters (defaults) |
| --- | --- |
| `IonotropicSynapse` | `IonotropicSynapse_gS` 1e-4, `_e_syn` 0, `_k_minus` 0.025, `_v_th` -35, `_delta` 10; state `_s` |
| `TanhConductanceSynapse` (0.11.0) | `_gS` 1e-4, `_e_syn` 0, `_x_offset` -70, `_slope` 1 |
| `TanhRateSynapse` | rate-based coupling for non-spiking networks |
| `TestSynapse` | minimal example for custom synapses |

```python
jx.connect(pre_view, post_view, synapse)                                   # one synapse
jx.fully_connect(pre_cells, post_cells, synapse, random_post_comp=False)
jx.sparse_connect(pre_cells, post_cells, synapse, p, random_post_comp=False)
jx.connectivity_matrix_connect(pre_cells, post_cells, synapse, bool_matrix, random_post_comp=False)
net.record("i_IonotropicSynapse")                                          # synaptic currents (0.6.0+)
```

## Optimisation helpers (`jaxley.optimize`)

```python
import jaxley.optimize.transforms as jt
jt.SigmoidTransform(lower, upper)      jt.LogisticTransform(lower, upper)   # 0.14.0
jt.SoftplusTransform(lower, threshold=20.0)   jt.NegSoftplusTransform(upper)
jt.AffineTransform(scale, shift)       jt.IdentityTransform()               # 0.14.0
jt.ChainTransform([t1, t2])            jt.MaskedTransform(mask, transform)
jt.CustomTransform(forward_fn, inverse_fn)

transform = jx.ParamTransform([{"HH_gNa": jt.SigmoidTransform(0.01, 0.5)}, ...])  # mirrors get_parameters()
opt_params = transform.inverse(params); params = transform.forward(opt_params)

from jaxley.optimize.utils import l2_norm          # L2 norm of a pytree
from jaxley.optimize import TypeOptimizer          # per-parameter-type optax optimizers
```

`ParamTransform` also accepts a single transform for a flat array of parameters (the L5PC example
uses `SigmoidTransform(lower_vector, upper_vector)` directly on a 20-vector with `data_set`).

## Morphology and I/O

- `jaxley.morphology.distance_direct(origin_view, cell)`, `distance_pathwise(origin_view, cell)`:
  Euclidean / along-the-tree distances of every compartment (for distance-dependent parameters).
- `jaxley.morphology.morph_delete(view)`, `morph_connect(view_a, view_b)`: edit or join morphologies.
- `jaxley.io.graph`: `swc_to_graph`, `from_graph(graph, ncomp=...)`, and module `.to_graph()`
  export for `networkx`-based editing.
- Pickle a module or use the "Save and load" how-to for checkpoints of the model itself; fitted
  parameters are better stored as data (this skill's `fitted_model.json`).

## Verified gotchas

- `write_trainables` raises `ValueError: Must have equal len keys and value` when two trainable
  entries share a parameter name (e.g. `Leak_gLeak` on two groups); `jx.integrate` handles the
  same parameters fine. Write values back per view with `set` instead.
- `make_trainable`, `record`, and `stimulate` print a line each unless `verbose=False`.
- `cell.set("length", x)` on a multi-compartment branch sets every compartment to `x`; the branch
  becomes `ncomp * x` long.
- A stimulus with the wrong number of samples does **not** raise: with `t_max` given, a shorter
  current is zero-padded and a longer one truncated, silently (verified in 0.14.0). Build every
  waveform on the protocol grid, `len(jx.step_current(0, 0, 0, delta_t, t_max))` samples;
  `_common.load_protocol` checks this for `currents_file` inputs.
- JAX >= 0.6 no longer needs `XLA_FLAGS=--xla_cpu_use_thunk_runtime=false`; the CPU slowdown that
  motivated Jaxley's old JAX pin is gone.
