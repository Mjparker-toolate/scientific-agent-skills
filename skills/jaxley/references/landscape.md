# Training nervous-system simulation models: the 2026 landscape

Surveyed 2026-09-17 to choose the scope of this skill. Release dates come from PyPI and GitHub;
"training support" means built-in gradient-based parameter fitting or task training, not merely
the ability to run a simulation inside an external optimiser.

## Differentiable biophysical simulation

| Tool | Latest release | What it is | Training support | Notes |
| --- | --- | --- | --- | --- |
| **Jaxley** ([jaxleyverse/jaxley](https://github.com/jaxleyverse/jaxley)) | 0.14.0, 2026-09-03 | Multicompartment conductance-based neurons and networks in JAX; SWC import; ion diffusion; four voltage solvers | Native: every simulation is differentiable; `make_trainable`, `ParamTransform`, `checkpoint_lengths`, `vmap` batching; optax training loops in the docs | Paper: Deistler et al., *Nature Methods* 2025. Pure Python, pip wheel, CPU/GPU/TPU. Chosen for this skill. |
| **BrainPy / brainpy.state + brainstate** ([chaobrain](https://github.com/chaobrain/brainpy.state)) | brainpy 2.8.2 (2026-07-23), brainstate 0.5.4 (2026-08-18) | Point-neuron spiking networks in JAX with physical units (`brainunit`), NEST-compatible models, surrogate-gradient BPTT, online learning (`braintrace`) | Native for point-neuron SNNs (`brainstate.transform.grad`, `braintools.optim`) | Strong for large point-neuron networks; no multicompartment morphologies. Ecosystem spread over several packages. |

## Spiking-network deep-learning frameworks (surrogate gradients on PyTorch)

| Tool | Latest release | Training support | Notes |
| --- | --- | --- | --- |
| **snnTorch** ([jeshraghian/snntorch](https://github.com/jeshraghian/snntorch)) | 1.0.0, 2026-06-29 | LIF variants as PyTorch layers, surrogate gradients, rate/latency losses | Neuromorphic ML, not biophysics; ideal for MNIST/N-MNIST-style tasks. |
| **SpikingJelly** ([fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)) | PyPI 0.0.0.0.14 (2023); GitHub active (pushed 2026-09-14) | Large-scale SNN training, ANN-to-SNN conversion, CuPy/Triton kernels, memory-efficient BPTT | Install from GitHub for current features. |
| **Norse** ([norse/norse](https://github.com/norse/norse)) | 1.1.0, 2024-03-18 | Functional PyTorch neurons, `torch.compile`-friendly | Low release cadence. |

These train networks to perform tasks; their neurons are abstractions (LIF, ALIF) rather than
conductance-based models fitted to physiology. Useful for brain-inspired computing, out of scope
for fitting nervous-system simulations to recordings.

## Connectome-constrained whole-system models

| Tool | Latest release | Training support | Notes |
| --- | --- | --- | --- |
| **flyvis** ([TuragaLab/flyvis](https://github.com/TuragaLab/flyvis)) | 1.2.0, 2026-08-06 | PyTorch deep mechanistic network of the fly visual system; trained on optic-flow estimation (Lappalainen et al., *Nature* 2024); CLI for training/ensembles | One lab's model of one system with fixed connectome and task; training needs the Sintel dataset and GPU ensembles. A skill would document a specific codebase, not a reusable method. |
| **FlyWire whole-brain LIF models** (Shiu et al. 2024; community repos) | data v783 | Activation/silencing experiments in Brian2; no parameter training | Simulation, not fitting. |
| **OpenWorm / c302** ([openworm/c302](https://github.com/openworm/c302)) | OpenWorm 0.9.8 (2026-03-24), c302 v0.12.0 (2026-03-31) | NeuroML2 generator for the *C. elegans* connectome, run via jNeuroML/NEURON | Model construction and export; parameter tuning is external (e.g. NeuroML tuning tools). |

## Classic simulators and their fitting workflows

| Tool | Latest release | Fitting workflow | Notes |
| --- | --- | --- | --- |
| **NEURON** | 9.0.2, 2026-08-10 | Via BluePyOpt, NetPyNE batch tools, or custom optimisers | Reference multicompartment simulator; NMODL mechanisms compiled with `nrnivmodl`. |
| **BluePyOpt** ([openbraininstitute/BluePyOpt](https://github.com/openbraininstitute/BluePyOpt)) | 1.14.25, 2026-08-24 | Evolutionary (IBEA/DEAP) multi-objective optimisation of eFEL features; NEURON and Arbor back ends | Maintained by Open Brain Institute since Blue Brain ended (Dec 2024). Gradient-free, needs compiled mechanisms; the standard for e-feature-based fits. |
| **NetPyNE** | 1.1.1, 2025-09-12 | `Batch` grid search, evolutionary (`evolOptim`), Optuna (`optunaOptim`) over NEURON networks | Network construction and exploration, not gradient training. |
| **Brian2** | 2.10.1, 2025-12-05 | `brian2modelfitting` toolbox (last release 0.4, 2020: Nevergrad / scikit-optimize / SBI, gradient-free) | Equation-based point/multicompartment simulator, code generation; no autodiff. |
| **NEST** | 3.10.0, 2026-06-10 | External optimisers; NESTML for models | Large-scale point-neuron networks and plasticity; no autodiff. |
| **Arbor** | 0.12.2, 2026-05-19 | BluePyOpt back end | GPU/HPC multicompartment simulator; no autodiff. Note: this repository's `arbor` skill is unrelated (the Arbor hypothesis-tree optimiser paper). |

## Why Jaxley for this skill

1. **It is the training pipeline for nervous-system simulation, in one package.** Differentiable
   simulation is what turns fitting a conductance-based model into gradient descent; Jaxley is the
   only maintained, pip-installable simulator that provides it for multicompartment biophysical
   models and networks, and it is documented for exactly that use (tutorial 07, the L5PC example,
   the *Nature Methods* paper's data- and task-constrained fits).
2. **Scope rule.** One package, one workflow - not an orchestrator over simulators, not a second
   provider for a covered service. No existing skill covers biophysical simulation or fitting
   (`neuropixels-analysis` is spike sorting, `neurokit2` is biosignals).
3. **Testable here.** Pure Python on CPU: the bundled scripts and tests run the full
   simulate -> fit -> validate loop in under a minute without compiled mechanisms (NEURON/Arbor),
   dataset downloads (flyvis), or a GPU (SpikingJelly-scale training).
4. **Current.** 0.14.0 shipped two weeks before this survey; the API used here was verified
   against that release by execution.

BrainPy/brainstate was the closest alternative (differentiable, JAX, actively released) but targets
point-neuron networks and spreads across several packages; it would make a good separate skill for
large spiking-network training. BluePyOpt remains the right tool when a legacy NEURON model with
NMODL mechanisms must be fitted without porting - Jaxley's NMODL import how-to is the bridge from
there to gradient-based fitting.
