"""Shared helpers for the Jaxley training-pipeline scripts.

A model and a stimulation protocol are plain JSON documents, so a fitted model
can be re-simulated, evaluated on held-out protocols, and version-controlled
without any Python state. Everything here is a thin layer over public Jaxley
0.14 APIs; nothing reaches into private attributes.

Model specification (see ``assets/hh_soma.json`` and ``assets/ball_and_stick.json``)::

    {
      "name": "hh_soma",
      "morphology": {"type": "single_compartment"},
      "v_init": -65.0,
      "channels":   [{"channel": "HH", "group": "all"}],
      "parameters": [{"key": "radius", "value": 10.0, "group": "all"}],
      "trainable":  [{"key": "HH_gNa", "group": "all", "lower": 0.01, "upper": 0.5}]
    }

``morphology.type`` is ``single_compartment``, ``ball_and_stick`` (soma + one
dendrite, groups ``soma`` and ``dendrite``) or ``swc`` (``path`` relative to the
spec file, optional ``ncomp`` and ``d_lambda`` block; groups ``soma``, ``axon``,
``basal``, ``apical`` come from the SWC type ids). ``channel`` is a class name in
``jaxley.channels`` or a dotted import path such as
``jaxley_mech.channels.l5pc.NaTs2T``. ``group`` is ``all`` or a group name.
A trainable entry with both bounds is optimised through a sigmoid, with only a
``lower`` bound through a softplus, with only an ``upper`` bound through a
negative softplus, and otherwise untransformed. ``per_compartment: true`` gives
every compartment of the group its own parameter.

Protocol specification (see ``assets/step_protocol.json``)::

    {
      "delta_t": 0.025, "t_max": 60.0, "solver": "bwd_euler",
      "stimulus":   {"group": "soma", "branch": 0, "loc": 0.5},
      "recordings": [{"group": "soma", "branch": 0, "loc": 0.5, "state": "v"}],
      "steps": [{"i_delay": 5.0, "i_dur": 40.0, "i_amp": 0.1}, ...]
    }

Each ``steps`` entry is one trial (one step current, nA). ``currents_file`` may
replace ``steps`` with an ``.npy``/``.npz`` (key ``currents``) or ``.csv`` array
of shape ``(n_trials, n_time)`` holding arbitrary waveforms.
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jaxley as jx
    import jaxley.channels
    import jaxley.optimize.transforms as jt
    import optax
    from jaxley.optimize.utils import l2_norm
except ImportError as error:  # pragma: no cover - exercised only without the packages
    raise SystemExit(
        "jaxley is not installed (or optax/jax is missing). "
        "Install the pipeline dependencies with `pip install jaxley optax`."
    ) from error

DEFAULT_V_INIT = -70.0
SPIKE_THRESHOLD_MV = 0.0
SOFT_SPIKE_WIDTH_MV = 2.0


# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #
def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def configure_jax(x64: bool = True, platform: str | None = None) -> None:
    """Enable float64 (the Jaxley tutorials' default) and optionally pin a platform.

    Must run before the first JAX array is created. float64 keeps the backward
    Euler solve and the gradients accurate for stiff channel kinetics; disable it
    only when trading accuracy for GPU throughput deliberately.
    """
    jax.config.update("jax_enable_x64", bool(x64))
    if platform:
        jax.config.update("jax_platform_name", platform)


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def resolve_channel(name: str):
    """Instantiate a channel by ``jaxley.channels`` class name or dotted path."""
    if "." in name:
        module_name, _, class_name = name.rpartition(".")
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()
    if not hasattr(jaxley.channels, name):
        available = sorted(
            attr for attr in dir(jaxley.channels) if attr[:1].isupper() and attr != "Channel"
        )
        raise ValueError(f"unknown channel {name!r}; built-ins are {available}")
    return getattr(jaxley.channels, name)()


def group_view(cell, group: str | None):
    """The whole module for ``all``/``None``, otherwise the named group view."""
    if group in (None, "all", ""):
        return cell
    if group not in cell.group_names:
        raise ValueError(f"group {group!r} does not exist; available groups: {cell.group_names}")
    return getattr(cell, group)


def site_view(cell, site: dict[str, Any]):
    """A single-location view, e.g. ``cell.soma.branch(0).loc(0.5)``."""
    view = group_view(cell, site.get("group"))
    return view.branch(int(site.get("branch", 0))).loc(float(site.get("loc", 0.5)))


def apply_d_lambda(cell, frequency: float = 100.0, d_lambda: float = 0.1) -> list[int]:
    """Set each branch's compartment count with NEURON's d_lambda rule.

    Mirrors the Jaxley how-to guide: the AC length constant at ``frequency`` Hz
    is computed from each branch's diameter, capacitance, and axial resistivity,
    and the branch is split into an odd number of compartments no longer than
    ``d_lambda`` length constants. Call on a cell read with ``ncomp=1`` so the
    branch length equals the single compartment's length.
    """
    counts = []
    for branch in cell.branches:
        nodes = branch.nodes
        diameter = 2.0 * float(nodes["radius"].to_numpy()[0])
        c_m = float(nodes["capacitance"].to_numpy()[0])
        r_a = float(nodes["axial_resistivity"].to_numpy()[0])
        length = float(nodes["length"].to_numpy().sum())
        lambda_f = 1e5 * math.sqrt(diameter / (4.0 * math.pi * frequency * c_m * r_a))
        ncomp = int((length / (d_lambda * lambda_f) + 0.9) / 2) * 2 + 1
        branch.set_ncomp(ncomp, initialize=False)
        counts.append(ncomp)
    cell.initialize()
    return counts


def _build_morphology(morph: dict[str, Any], base_dir: Path):
    kind = morph.get("type", "single_compartment")
    comp = jx.Compartment()
    if kind == "single_compartment":
        cell = jx.Cell(jx.Branch(comp, ncomp=1), parents=[-1])
        cell.add_to_group("soma")
        return cell
    if kind == "ball_and_stick":
        soma = jx.Branch(comp, ncomp=1)
        dendrite = jx.Branch(comp, ncomp=int(morph.get("dendrite_ncomp", 4)))
        cell = jx.Cell([soma, dendrite], parents=[-1, 0])
        cell.branch(0).add_to_group("soma")
        cell.branch(1).add_to_group("dendrite")
        cell.soma.set("radius", float(morph.get("soma_radius", 10.0)))
        cell.soma.set("length", float(morph.get("soma_length", 20.0)))
        cell.dendrite.set("radius", float(morph.get("dendrite_radius", 1.0)))
        # `length` is per compartment; the spec gives the whole dendrite.
        cell.dendrite.set(
            "length", float(morph.get("dendrite_length", 200.0)) / int(morph.get("dendrite_ncomp", 4))
        )
        return cell
    if kind == "swc":
        path = Path(morph["path"])
        if not path.is_absolute():
            path = base_dir / path
        cell = jx.read_swc(
            str(path),
            ncomp=int(morph.get("ncomp", 1)),
            min_radius=morph.get("min_radius"),
            assign_groups=True,
        )
        if "d_lambda" in morph:
            rule = morph["d_lambda"] or {}
            apply_d_lambda(
                cell,
                frequency=float(rule.get("frequency", 100.0)),
                d_lambda=float(rule.get("d_lambda", 0.1)),
            )
        return cell
    raise ValueError(f"unknown morphology type {kind!r}")


def build_model(spec: dict[str, Any], base_dir: str | Path | None = None):
    """Build a ``jx.Cell`` from a model specification (see module docstring)."""
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    cell = _build_morphology(spec.get("morphology", {}), base)
    for entry in spec.get("channels", []):
        group_view(cell, entry.get("group")).insert(resolve_channel(entry["channel"]))
    apply_parameters(cell, spec.get("parameters", []))
    cell.set("v", float(spec.get("v_init", DEFAULT_V_INIT)))
    cell.init_states()
    return cell


def apply_parameters(cell, entries: list[dict[str, Any]]) -> None:
    """``set`` each ``{"key", "value", "group", "per_compartment"}`` entry."""
    for entry in entries:
        view = group_view(cell, entry.get("group"))
        value = entry["value"]
        if entry.get("per_compartment") or isinstance(value, list):
            view = view.branch("all").comp("all")
            value = np.asarray(value, dtype=float)
            if value.size != len(view.nodes):
                raise ValueError(
                    f"{entry['key']}: {value.size} values for {len(view.nodes)} compartments"
                )
        view.set(entry["key"], value)


def make_transform(entry: dict[str, Any]):
    lower, upper = entry.get("lower"), entry.get("upper")
    if lower is not None and upper is not None:
        if not lower < upper:
            raise ValueError(f"{entry['key']}: lower bound {lower} must be below upper bound {upper}")
        return jt.SigmoidTransform(float(lower), float(upper))
    if lower is not None:
        return jt.SoftplusTransform(float(lower))
    if upper is not None:
        return jt.NegSoftplusTransform(float(upper))
    return jt.IdentityTransform()


def add_trainables(cell, spec: dict[str, Any]):
    """Register the spec's trainables on ``cell`` and return the ``ParamTransform``.

    ``cell.get_parameters()`` afterwards returns one dict per trainable entry, in
    spec order, which is exactly the structure the transform expects.
    """
    entries = spec.get("trainable", [])
    if not entries:
        raise ValueError("the model spec declares no `trainable` parameters")
    cell.delete_trainables()
    transforms = []
    for entry in entries:
        view = group_view(cell, entry.get("group"))
        if entry.get("per_compartment"):
            view = view.branch("all").comp("all")
        view.make_trainable(entry["key"], verbose=False)
        transforms.append({entry["key"]: make_transform(entry)})
    return jx.ParamTransform(transforms)


def parameters_to_records(spec: dict[str, Any], params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair ``get_parameters()`` output with the spec's trainable entries."""
    records = []
    for entry, param in zip(spec.get("trainable", []), params, strict=True):
        values = np.asarray(param[entry["key"]], dtype=float).ravel()
        record = {"key": entry["key"], "group": entry.get("group", "all")}
        if entry.get("per_compartment") or values.size > 1:
            record["per_compartment"] = True
            record["value"] = values.tolist()
        else:
            record["value"] = float(values[0])
        records.append(record)
    return records


def spec_with_parameters(spec: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """A copy of ``spec`` whose ``parameters`` list realises the fitted values.

    Later entries win in ``apply_parameters``, so appending is enough; the
    ``trainable`` block is kept so the model can be fitted further.
    """
    fitted = json.loads(json.dumps(spec))
    fitted["parameters"] = list(spec.get("parameters", [])) + [dict(record) for record in records]
    return fitted


def sample_within_bounds(spec: dict[str, Any], params, n_samples: int, seed: int):
    """Uniform samples inside every trainable's bounds, as a stacked pytree.

    Random-search initialisation before gradient descent is what the Jaxley
    L5PC example recommends; it needs finite bounds on every trainable.
    """
    rng = np.random.default_rng(seed)
    stacked = []
    for entry, param in zip(spec.get("trainable", []), params, strict=True):
        if entry.get("lower") is None or entry.get("upper") is None:
            raise ValueError(
                f"{entry['key']}: random initialisation needs both `lower` and `upper` bounds"
            )
        shape = (n_samples,) + tuple(np.shape(param[entry["key"]]))
        draw = rng.uniform(float(entry["lower"]), float(entry["upper"]), size=shape)
        stacked.append({entry["key"]: jnp.asarray(draw)})
    return stacked


# --------------------------------------------------------------------------- #
# Protocols and simulation
# --------------------------------------------------------------------------- #
@dataclass
class Protocol:
    delta_t: float
    t_max: float
    currents: np.ndarray  # (n_trials, n_time), nA
    stimulus: dict[str, Any]
    recordings: list[dict[str, Any]]
    solver: str = "bwd_euler"
    labels: list[str] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return int(self.currents.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.currents.shape[1])

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.n_time) * self.delta_t


def n_time_points(delta_t: float, t_max: float) -> int:
    """Length of the traces ``jx.integrate`` returns for ``t_max`` at ``delta_t``."""
    return int(len(jx.step_current(0.0, 0.0, 0.0, delta_t, t_max)))


def load_protocol(path: str | Path) -> Protocol:
    path = Path(path)
    spec = load_json(path)
    delta_t = float(spec.get("delta_t", 0.025))
    t_max = float(spec["t_max"])
    n_time = n_time_points(delta_t, t_max)
    if "steps" in spec:
        rows, labels = [], []
        for index, step in enumerate(spec["steps"]):
            rows.append(
                np.asarray(
                    jx.step_current(
                        float(step.get("i_delay", 0.0)),
                        float(step["i_dur"]),
                        float(step["i_amp"]),
                        delta_t,
                        t_max,
                        float(step.get("i_offset", 0.0)),
                    )
                )
            )
            labels.append(step.get("label", f"step_{index}_{step['i_amp']:g}nA"))
        currents = np.stack(rows)
    elif "currents_file" in spec:
        currents = load_array(path.parent / spec["currents_file"], key="currents")
        currents = np.atleast_2d(np.asarray(currents, dtype=float))
        if currents.shape[1] != n_time:
            raise ValueError(
                f"currents_file has {currents.shape[1]} samples per trial, but "
                f"t_max={t_max} at delta_t={delta_t} needs {n_time}"
            )
        labels = [f"trial_{index}" for index in range(currents.shape[0])]
    else:
        raise ValueError("protocol needs either `steps` or `currents_file`")
    recordings = spec.get("recordings") or [dict(spec.get("stimulus", {}), state="v")]
    return Protocol(
        delta_t=delta_t,
        t_max=t_max,
        currents=currents,
        stimulus=spec.get("stimulus", {}),
        recordings=recordings,
        solver=spec.get("solver", "bwd_euler"),
        labels=labels,
    )


def load_array(path: str | Path, key: str = "traces") -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            if key not in archive:
                raise KeyError(f"{path} has no array named {key!r}; found {list(archive.keys())}")
            return np.asarray(archive[key])
    if path.suffix == ".npy":
        return np.load(path)
    return np.loadtxt(path, delimiter=",")


def attach_recordings(cell, protocol: Protocol) -> None:
    cell.delete_recordings()
    for recording in protocol.recordings:
        site_view(cell, recording).record(recording.get("state", "v"), verbose=False)


def checkpoint_lengths(n_time: int, levels: int) -> list[int] | None:
    """Multi-level checkpoint lengths, as in the Jaxley training tutorial.

    ``levels=0`` disables checkpointing (fastest, most memory). Two levels bring
    the memory of the backward pass from O(T) down to O(sqrt(T)).
    """
    if levels <= 0:
        return None
    return [int(math.ceil(n_time ** (1.0 / levels)))] * levels


def make_simulator(cell, protocol: Protocol, checkpoint_levels: int = 0):
    """Return ``simulate(params, currents) -> (n_trials, n_recordings, n_time)``.

    The stimulus is injected with ``data_stimulate`` so the current is a traced
    argument and the function can be ``vmap``-ed over trials and differentiated
    with respect to ``params``.
    """
    attach_recordings(cell, protocol)
    lengths = checkpoint_lengths(protocol.n_time, checkpoint_levels)
    stimulus = protocol.stimulus

    def simulate_one(params, current):
        data_stimuli = site_view(cell, stimulus).data_stimulate(current, data_stimuli=None)
        return jx.integrate(
            cell,
            params=params,
            data_stimuli=data_stimuli,
            delta_t=protocol.delta_t,
            t_max=protocol.t_max,
            solver=protocol.solver,
            checkpoint_lengths=lengths,
        )

    return jax.vmap(simulate_one, in_axes=(None, 0))


# --------------------------------------------------------------------------- #
# Losses and summary statistics
# --------------------------------------------------------------------------- #
def mse_loss(prediction, target, start_index: int = 0):
    """Mean squared voltage error after ``start_index`` samples (mV^2)."""
    diff = prediction[..., start_index:] - target[..., start_index:]
    return jnp.mean(diff**2)


def summary_statistics(traces, n_windows: int = 4):
    """Per-window mean and standard deviation of every trace.

    Splits the time axis into ``n_windows`` equal windows and returns an array of
    shape ``(..., n_windows, 2)``. These are differentiable, unlike spike times,
    and are the summary statistics used in the Jaxley L5PC example.
    """
    n_time = traces.shape[-1]
    edges = np.linspace(0, n_time, n_windows + 1).astype(int)
    stats = []
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        window = traces[..., start:stop]
        stats.append(jnp.stack([jnp.mean(window, axis=-1), jnp.std(window, axis=-1)], axis=-1))
    return jnp.stack(stats, axis=-2)


def summary_loss(prediction, target, n_windows: int = 4, mean_scale: float = 2.0, std_scale: float = 1.0):
    """Standardised mean absolute error between summary statistics.

    ``mean_scale`` and ``std_scale`` (mV) down-weight the respective statistic;
    the defaults follow the L5PC example. Standardising is what makes a
    loss-scaled (Polyak) learning rate safe.
    """
    scale = jnp.asarray([mean_scale, std_scale])
    difference = summary_statistics(prediction, n_windows) - summary_statistics(target, n_windows)
    return jnp.mean(jnp.abs(difference / scale))


def soft_spike_count(traces, threshold: float = SPIKE_THRESHOLD_MV, width: float = SOFT_SPIKE_WIDTH_MV):
    """Differentiable count of upward threshold crossings along the last axis.

    The total positive variation of ``sigmoid((v - threshold) / width)``: each
    excursion from well below to well above the threshold and back contributes
    exactly one, however many samples the upstroke spans. For action potentials
    that overshoot the threshold by several ``width``s this matches the hard
    count to within a few percent; a spike that only grazes the threshold counts
    fractionally, which is what makes the loss smooth.
    """
    above = jax.nn.sigmoid((traces - threshold) / width)
    return jnp.sum(jax.nn.relu(above[..., 1:] - above[..., :-1]), axis=-1)


def spike_count_loss(prediction, target, threshold: float = SPIKE_THRESHOLD_MV, width: float = SOFT_SPIKE_WIDTH_MV):
    """Squared error between soft spike counts, averaged over trials/recordings."""
    return jnp.mean((soft_spike_count(prediction, threshold, width) - soft_spike_count(target, threshold, width)) ** 2)


LOSSES = {
    "mse": mse_loss,
    "summary": summary_loss,
    "spike_count": spike_count_loss,
}


def hard_spike_count(traces: np.ndarray, threshold: float = SPIKE_THRESHOLD_MV) -> np.ndarray:
    """Number of upward threshold crossings along the last axis (numpy, not differentiable)."""
    traces = np.asarray(traces)
    above = traces > threshold
    return np.sum(above[..., 1:] & ~above[..., :-1], axis=-1)


def spike_times(trace: np.ndarray, time: np.ndarray, threshold: float = SPIKE_THRESHOLD_MV) -> np.ndarray:
    """Times (ms) of upward threshold crossings of one trace."""
    trace = np.asarray(trace)
    above = trace > threshold
    crossings = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    return np.asarray(time)[crossings]


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def make_adam(learning_rate: float, clip_norm: float | None):
    """Adam with optional global-norm gradient clipping, in transformed space."""
    steps = []
    if clip_norm is not None and clip_norm > 0:
        steps.append(optax.clip_by_global_norm(clip_norm))
    steps.append(optax.adam(learning_rate))
    return optax.chain(*steps)


def polyak_step(grads, loss_value, mu: float, beta: float, eps: float = 1e-12):
    """Normalise the gradient and return ``(scaled_grads, learning_rate)``.

    The Polyak-style rule from the Jaxley L5PC example: divide the gradient by
    its norm to the power ``beta`` and scale the learning rate by the loss, so
    steps shrink as the fit improves. ``eps`` guards against a vanishing norm --
    once a sigmoid-bounded parameter saturates its gradient underflows and the
    unguarded division produces NaN.
    """
    norm = l2_norm(grads) + eps
    scaled = jax.tree_util.tree_map(lambda leaf: leaf / norm**beta, grads)
    return scaled, loss_value * mu


def finite(pytree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(pytree))


def saturation_report(opt_params, limit: float = 8.0) -> list[str]:
    """Names of transformed parameters pinned near a bound (|value| > ``limit``).

    In sigmoid space, |8| corresponds to being within 0.03% of the bound: the
    gradient there is essentially zero and the parameter can no longer move.
    """
    pinned = []
    for group in opt_params:
        for key, value in group.items():
            if bool(jnp.any(jnp.abs(value) > limit)):
                pinned.append(key)
    return pinned
