"""Tests for the Jaxley training-pipeline scripts.

The shared helper (`_common`) does the work an agent would otherwise re-derive:
turning a JSON model spec into a `jx.Cell`, registering bounded trainables,
building a batched differentiable simulator, and the loss functions. Those
pieces are checked against values fixed by the spec files and by Jaxley's own
defaults (HH_gNa = 0.12 S/cm^2 and friends), so a mis-scoped `group`, a
transform that leaks outside its bounds, or a spike counter off by one fails
here rather than in a fit.

The three CLIs are then driven in-process on the bundled assets: the synthetic
round trip that SKILL.md documents has to hold -- simulate a ground truth,
perturb the conductances by 50%, recover them by gradient descent, and score
the fit on a protocol the fit never saw. Random-start screening, parallel
top-k descent, both optimizers, checkpointing and the failure gate of
`evaluate_fit.py` each get one small real run.

Everything here needs jaxley + optax; the suite skips cleanly without them
and runs for real under `python tests/run_all.py --isolated jaxley`.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

import skill_contract

SKILL_ROOT = Path(__file__).resolve().parents[2] / "skills" / "jaxley"
SCRIPTS = SKILL_ROOT / "scripts"
ASSETS = SKILL_ROOT / "assets"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("jax", reason="jaxley needs jax")
pytest.importorskip("jaxley", reason="the pipeline scripts need jaxley")
pytest.importorskip("optax", reason="the pipeline scripts need optax")

import _common as common  # noqa: E402
import evaluate_fit  # noqa: E402
import fit_biophysics  # noqa: E402
import simulate_protocol  # noqa: E402

common.configure_jax(x64=True, platform="cpu")
jnp = common.jnp

CliHelpTests = skill_contract.cli.help_test_case(SKILL_ROOT)

#: Jaxley's HH defaults, which hh_soma.json spells out as its ground truth.
HH_TRUTH = {"HH_gNa": 0.12, "HH_gK": 0.036, "HH_gLeak": 0.0003}


def quiet(function, *args, **kwargs):
    """Run a CLI `main` with its stdout captured; return (exit code, output)."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = function(*args, **kwargs)
    return code, stream.getvalue()


class ModelSpecTests(unittest.TestCase):
    def test_the_single_compartment_asset_builds_as_specified(self) -> None:
        spec = common.load_json(ASSETS / "hh_soma.json")
        cell = common.build_model(spec, base_dir=ASSETS)
        self.assertEqual(len(cell.nodes), 1)
        self.assertEqual(cell.group_names, ["soma"])
        row = cell.nodes.iloc[0]
        self.assertTrue(bool(row["HH"]))
        self.assertAlmostEqual(row["radius"], 10.0)
        self.assertAlmostEqual(row["length"], 20.0)
        self.assertAlmostEqual(row["axial_resistivity"], 100.0)
        for key, value in HH_TRUTH.items():
            self.assertAlmostEqual(row[key], value)
        # init_states() ran: the gating variables left Jaxley's 0.2 placeholders.
        self.assertNotAlmostEqual(row["HH_m"], 0.2)
        self.assertAlmostEqual(row["v"], -65.0)

    def test_the_ball_and_stick_asset_scopes_channels_and_geometry_by_group(self) -> None:
        spec = common.load_json(ASSETS / "ball_and_stick.json")
        cell = common.build_model(spec, base_dir=ASSETS)
        self.assertEqual(len(cell.nodes), 5)
        self.assertEqual(sorted(cell.group_names), ["dendrite", "soma"])
        self.assertEqual(len(cell.soma.nodes), 1)
        self.assertEqual(len(cell.dendrite.nodes), 4)
        # HH only in the soma, Leak only in the dendrite.
        self.assertTrue(bool(cell.soma.nodes["HH"].all()))
        self.assertFalse(bool(cell.dendrite.nodes["HH"].any()))
        self.assertTrue(bool(cell.dendrite.nodes["Leak"].all()))
        self.assertFalse(bool(cell.soma.nodes["Leak"].any()))
        # The spec gives the whole dendrite length; each compartment gets a quarter.
        np.testing.assert_allclose(cell.dendrite.nodes["length"].to_numpy(), 50.0)
        np.testing.assert_allclose(cell.dendrite.nodes["radius"].to_numpy(), 1.0)
        np.testing.assert_allclose(cell.dendrite.nodes["Leak_eLeak"].to_numpy(), -65.0)

    def test_an_swc_morphology_is_read_relative_to_the_spec_and_grouped(self) -> None:
        spec = {
            "morphology": {"type": "swc", "path": "tiny.swc", "ncomp": 2},
            "channels": [{"channel": "Leak", "group": "all"}, {"channel": "HH", "group": "soma"}],
        }
        cell = common.build_model(spec, base_dir=FIXTURES)
        # Four branches (soma, trunk, two daughters) x two compartments.
        self.assertEqual(len(cell.nodes), 8)
        self.assertIn("soma", cell.group_names)
        self.assertIn("basal", cell.group_names)
        self.assertEqual(len(cell.soma.nodes), 2)
        self.assertEqual(len(cell.basal.nodes), 6)
        self.assertTrue(bool(cell.soma.nodes["HH"].all()))
        self.assertFalse(bool(cell.basal.nodes["HH"].any()))

    def test_the_d_lambda_rule_gives_every_branch_an_odd_compartment_count(self) -> None:
        spec = {
            "morphology": {
                "type": "swc",
                "path": "tiny.swc",
                "ncomp": 1,
                "d_lambda": {"frequency": 100.0, "d_lambda": 0.1},
            },
            "parameters": [{"key": "axial_resistivity", "value": 100.0, "group": "all"}],
        }
        cell = common.build_model(spec, base_dir=FIXTURES)
        counts = cell.nodes.groupby("global_branch_index").size().tolist()
        self.assertEqual(len(counts), 4)
        for count in counts:
            self.assertEqual(count % 2, 1)
        # The thin 200 um trunk needs finer spatial resolution than the 10 um soma.
        self.assertGreater(counts[1], counts[0])

    def test_unknown_channels_and_groups_fail_with_a_helpful_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown channel 'NotAChannel'.*HH"):
            common.resolve_channel("NotAChannel")
        cell = common.build_model({"morphology": {"type": "single_compartment"}})
        with self.assertRaisesRegex(ValueError, "group 'axon' does not exist.*soma"):
            common.group_view(cell, "axon")
        with self.assertRaisesRegex(ValueError, "unknown morphology type"):
            common.build_model({"morphology": {"type": "cube"}})

    def test_a_dotted_channel_path_is_imported(self) -> None:
        channel = common.resolve_channel("jaxley.channels.Leak")
        self.assertEqual(type(channel).__name__, "Leak")


class TransformAndTrainableTests(unittest.TestCase):
    def test_bounded_transforms_stay_inside_their_bounds(self) -> None:
        transform = common.make_transform({"key": "g", "lower": 0.01, "upper": 0.5})
        extreme = jnp.asarray([-50.0, -3.0, 0.0, 3.0, 50.0])
        values = np.asarray(transform.forward(extreme))
        self.assertTrue(np.all(values >= 0.01))
        self.assertTrue(np.all(values <= 0.5))
        np.testing.assert_allclose(np.asarray(transform.inverse(transform.forward(extreme[1:4]))), extreme[1:4])

    def test_one_sided_and_unbounded_entries_pick_the_matching_transform(self) -> None:
        lower_only = common.make_transform({"key": "g", "lower": 0.0})
        self.assertGreater(float(lower_only.forward(jnp.asarray(-40.0))), 0.0)
        upper_only = common.make_transform({"key": "g", "upper": 1.0})
        self.assertLess(float(upper_only.forward(jnp.asarray(40.0))), 1.0)
        identity = common.make_transform({"key": "g"})
        self.assertAlmostEqual(float(identity.forward(jnp.asarray(2.5))), 2.5)
        with self.assertRaisesRegex(ValueError, "lower bound"):
            common.make_transform({"key": "g", "lower": 1.0, "upper": 0.5})

    def test_trainables_follow_the_spec_order_and_round_trip_into_a_spec(self) -> None:
        spec = common.load_json(ASSETS / "ball_and_stick.json")
        cell = common.build_model(spec, base_dir=ASSETS)
        transform = common.add_trainables(cell, spec)
        params = cell.get_parameters()
        self.assertEqual([list(group) for group in params], [["HH_gNa"], ["HH_gK"], ["Leak_gLeak"], ["radius"]])
        np.testing.assert_allclose(np.asarray(transform.forward(transform.inverse(params))[3]["radius"]), 1.0)

        records = common.parameters_to_records(spec, params)
        self.assertEqual([record["group"] for record in records], ["soma", "soma", "dendrite", "dendrite"])
        self.assertAlmostEqual(records[0]["value"], 0.12)
        self.assertAlmostEqual(records[3]["value"], 1.0)

        # Writing changed values back through the spec must land on the right group.
        records[2]["value"] = 4e-4
        records[3]["value"] = 2.0
        rebuilt = common.build_model(common.spec_with_parameters(spec, records), base_dir=ASSETS)
        np.testing.assert_allclose(rebuilt.dendrite.nodes["Leak_gLeak"].to_numpy(), 4e-4)
        np.testing.assert_allclose(rebuilt.dendrite.nodes["radius"].to_numpy(), 2.0)
        np.testing.assert_allclose(rebuilt.soma.nodes["radius"].to_numpy(), 10.0)

    def test_per_compartment_trainables_give_one_value_per_compartment(self) -> None:
        spec = common.load_json(ASSETS / "ball_and_stick.json")
        spec["trainable"] = [{"key": "Leak_gLeak", "group": "dendrite", "per_compartment": True, "lower": 1e-5, "upper": 1e-3}]
        cell = common.build_model(spec, base_dir=ASSETS)
        common.add_trainables(cell, spec)
        params = cell.get_parameters()
        self.assertEqual(params[0]["Leak_gLeak"].shape, (4,))
        records = common.parameters_to_records(spec, params)
        self.assertTrue(records[0]["per_compartment"])
        self.assertEqual(len(records[0]["value"]), 4)
        records[0]["value"] = [1e-4, 2e-4, 3e-4, 4e-4]
        rebuilt = common.build_model(common.spec_with_parameters(spec, records), base_dir=ASSETS)
        np.testing.assert_allclose(rebuilt.dendrite.nodes["Leak_gLeak"].to_numpy(), records[0]["value"])

    def test_random_starts_are_uniform_inside_every_bound(self) -> None:
        spec = common.load_json(ASSETS / "hh_soma.json")
        cell = common.build_model(spec, base_dir=ASSETS)
        common.add_trainables(cell, spec)
        draws = common.sample_within_bounds(spec, cell.get_parameters(), n_samples=200, seed=3)
        for entry, group in zip(spec["trainable"], draws, strict=True):
            values = np.asarray(group[entry["key"]])
            self.assertEqual(values.shape, (200, 1))
            self.assertGreaterEqual(values.min(), entry["lower"])
            self.assertLessEqual(values.max(), entry["upper"])
        spec["trainable"][0].pop("upper")
        with self.assertRaisesRegex(ValueError, "needs both"):
            common.sample_within_bounds(spec, cell.get_parameters(), n_samples=2, seed=0)

    def test_a_spec_without_trainables_is_rejected(self) -> None:
        cell = common.build_model({"morphology": {"type": "single_compartment"}})
        with self.assertRaisesRegex(ValueError, "no `trainable`"):
            common.add_trainables(cell, {})


class ProtocolTests(unittest.TestCase):
    def test_step_protocols_become_one_current_row_per_trial(self) -> None:
        protocol = common.load_protocol(ASSETS / "step_protocol.json")
        self.assertEqual(protocol.currents.shape, (3, 2401))
        self.assertEqual(protocol.n_time, common.n_time_points(0.025, 60.0))
        self.assertAlmostEqual(protocol.time[-1], 60.0)
        self.assertEqual(protocol.labels, ["step_0.05nA", "step_0.10nA", "step_0.20nA"])
        np.testing.assert_allclose(protocol.currents.max(axis=1), [0.05, 0.1, 0.2])
        # 5 ms delay, 40 ms duration: on at index 200, off again by 45 ms.
        self.assertEqual(protocol.currents[1, 199], 0.0)
        self.assertEqual(protocol.currents[1, 201], 0.1)
        self.assertEqual(protocol.currents[1, 1810], 0.0)

    def test_a_currents_file_replaces_steps_and_must_match_t_max(self) -> None:
        currents = np.zeros((2, common.n_time_points(0.1, 10.0)))
        currents[0, 20:60] = 0.3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "currents.npz", currents=currents)
            (root / "protocol.json").write_text(json.dumps({
                "delta_t": 0.1, "t_max": 10.0, "currents_file": "currents.npz",
                "stimulus": {"group": "soma", "branch": 0, "loc": 0.5},
            }))
            protocol = common.load_protocol(root / "protocol.json")
            np.testing.assert_array_equal(protocol.currents, currents)
            # Recordings default to the stimulus site.
            self.assertEqual(protocol.recordings[0]["group"], "soma")
            self.assertEqual(protocol.recordings[0]["state"], "v")

            (root / "protocol.json").write_text(json.dumps({
                "delta_t": 0.1, "t_max": 20.0, "currents_file": "currents.npz",
                "stimulus": {"group": "soma"},
            }))
            with self.assertRaisesRegex(ValueError, "samples per trial"):
                common.load_protocol(root / "protocol.json")

    def test_a_protocol_needs_steps_or_a_currents_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps({"t_max": 5.0, "stimulus": {}}))
            with self.assertRaisesRegex(ValueError, "steps"):
                common.load_protocol(path)

    def test_checkpoint_lengths_follow_the_tutorial_formula(self) -> None:
        self.assertIsNone(common.checkpoint_lengths(2401, 0))
        # 49^2 = 2401 exactly; one more sample needs the next integer up.
        self.assertEqual(common.checkpoint_lengths(2401, 2), [49, 49])
        self.assertEqual(common.checkpoint_lengths(2402, 2), [50, 50])
        self.assertEqual(common.checkpoint_lengths(1000, 3), [10, 10, 10])


class LossTests(unittest.TestCase):
    def test_hard_and_soft_spike_counts_agree_on_square_pulses(self) -> None:
        trace = np.full(400, -70.0)
        for start in (50, 150, 300):
            trace[start : start + 10] = 30.0
        self.assertEqual(int(common.hard_spike_count(trace)), 3)
        self.assertAlmostEqual(float(common.soft_spike_count(jnp.asarray(trace))), 3.0, places=4)
        np.testing.assert_allclose(common.spike_times(trace, np.arange(400) * 0.1), [5.0, 15.0, 30.0])
        # A trace that never crosses the threshold has no spikes, softly or hard.
        self.assertEqual(int(common.hard_spike_count(np.full(100, -60.0))), 0)
        self.assertLess(float(common.soft_spike_count(jnp.full(100, -60.0))), 1e-6)

    def test_the_soft_count_tracks_the_hard_count_on_a_real_hh_trace(self) -> None:
        spec = common.load_json(ASSETS / "hh_soma.json")
        protocol = common.load_protocol(ASSETS / "step_protocol.json")
        cell = common.build_model(spec, base_dir=ASSETS)
        traces = common.make_simulator(cell, protocol)([], jnp.asarray(protocol.currents))
        hard = common.hard_spike_count(np.asarray(traces))
        soft = np.asarray(common.soft_spike_count(traces))
        self.assertTrue(np.all(hard[:, 0] >= 1))
        self.assertTrue(np.all(np.diff(hard[:, 0]) >= 0), "more current, more spikes")
        np.testing.assert_allclose(soft, hard, atol=0.15)

    def test_summary_statistics_have_one_mean_and_std_per_window(self) -> None:
        traces = jnp.asarray(np.random.default_rng(0).normal(size=(2, 1, 400)))
        stats = common.summary_statistics(traces, n_windows=4)
        self.assertEqual(stats.shape, (2, 1, 4, 2))
        np.testing.assert_allclose(np.asarray(stats[0, 0, 0, 0]), np.mean(np.asarray(traces[0, 0, :100])))
        self.assertAlmostEqual(float(common.summary_loss(traces, traces)), 0.0)
        self.assertGreater(float(common.summary_loss(traces, traces + 5.0)), 0.0)

    def test_the_mse_loss_can_skip_a_burn_in(self) -> None:
        prediction = jnp.zeros((1, 1, 10))
        target = jnp.concatenate([jnp.full((1, 1, 5), 10.0), jnp.zeros((1, 1, 5))], axis=-1)
        self.assertAlmostEqual(float(common.mse_loss(prediction, target)), 50.0)
        self.assertAlmostEqual(float(common.mse_loss(prediction, target, start_index=5)), 0.0)


class PipelineTests(unittest.TestCase):
    """The synthetic round trip from SKILL.md, on the bundled assets."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        cls.target = cls.root / "target.npz"
        cls.heldout = cls.root / "heldout.npz"
        code, cls.simulate_output = quiet(
            simulate_protocol.main,
            ["--model", str(ASSETS / "hh_soma.json"), "--protocol", str(ASSETS / "step_protocol.json"), "--out", str(cls.target)],
        )
        assert code == 0, cls.simulate_output
        code, _ = quiet(
            simulate_protocol.main,
            ["--model", str(ASSETS / "hh_soma.json"), "--protocol", str(ASSETS / "heldout_protocol.json"), "--out", str(cls.heldout)],
        )
        assert code == 0

        # The starting point of the fit: every conductance 50% too high.
        spec = common.load_json(ASSETS / "hh_soma.json")
        for entry in spec["parameters"]:
            if entry["key"] in HH_TRUTH:
                entry["value"] *= 1.5
        spec["name"] = "hh_soma_perturbed"
        cls.perturbed = cls.root / "perturbed.json"
        cls.perturbed.write_text(json.dumps(spec))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def fit(self, out: str, *extra: str) -> dict:
        code, output = quiet(
            fit_biophysics.main,
            [
                "--model", str(self.perturbed), "--protocol", str(ASSETS / "step_protocol.json"),
                "--target", str(self.target), "--out", str(self.root / out), "--log-every", "1000", *extra,
            ],
        )
        self.assertEqual(code, 0, output)
        return common.load_json(self.root / out / "fit_result.json")

    def test_simulation_writes_finite_traces_with_the_protocol_layout(self) -> None:
        with np.load(self.target) as archive:
            traces, time, currents, labels = archive["traces"], archive["time"], archive["currents"], archive["labels"]
        self.assertEqual(traces.shape, (3, 1, 2401))
        self.assertTrue(np.all(np.isfinite(traces)))
        self.assertEqual(time.shape, (2401,))
        self.assertEqual(currents.shape, (3, 2401))
        self.assertEqual(list(labels), ["step_0.05nA", "step_0.10nA", "step_0.20nA"])
        # Resting near -65 mV before the stimulus, overshooting past 0 mV during it.
        self.assertLess(abs(traces[0, 0, 0] + 65.0), 1.0)
        self.assertGreater(traces[:, 0, :].max(), 0.0)
        self.assertIn("spikes per recording", self.simulate_output)

    def test_the_truth_scores_itself_perfectly_on_the_held_out_protocol(self) -> None:
        report_path = self.root / "truth_report.json"
        code, _ = quiet(
            evaluate_fit.main,
            [
                "--model", str(ASSETS / "hh_soma.json"), "--protocol", str(ASSETS / "heldout_protocol.json"),
                "--target", str(self.heldout), "--out", str(report_path), "--max-rmse", "0.001", "--max-spike-count-error", "0",
            ],
        )
        self.assertEqual(code, 0)
        report = common.load_json(report_path)
        self.assertTrue(report["passed"])
        self.assertLess(report["mean_rmse_mV"], 1e-9)
        self.assertEqual(report["mean_abs_spike_count_error"], 0.0)
        self.assertEqual(len(report["trials"]), 2)
        self.assertEqual(report["trials"][0]["spikes_model"], report["trials"][0]["spikes_target"])
        self.assertEqual(report["trials"][0]["first_spike_latency_error_ms"], 0.0)

    def test_the_perturbed_model_fails_the_gate_and_the_metrics_say_why(self) -> None:
        report_path = self.root / "perturbed_report.json"
        code, output = quiet(
            evaluate_fit.main,
            [
                "--model", str(self.perturbed), "--protocol", str(ASSETS / "step_protocol.json"),
                "--target", str(self.target), "--out", str(report_path), "--max-rmse", "1.0",
            ],
        )
        self.assertEqual(code, 1)
        self.assertIn("FAIL", output)
        report = common.load_json(report_path)
        self.assertFalse(report["passed"])
        self.assertGreater(report["mean_rmse_mV"], 1.0)
        self.assertEqual(len(report["failures"]), 1)

    def test_gradient_descent_recovers_the_conductances_from_a_perturbed_start(self) -> None:
        result = self.fit("fit_spec", "--steps", "40", "--learning-rate", "0.1")
        self.assertEqual(result["n_parameters"], 3)
        self.assertEqual(result["n_candidates_screened"], 1)
        self.assertEqual(result["stop_reason"], "completed all steps")
        self.assertEqual(result["steps_run"], 40)
        self.assertLess(result["best_loss"], 0.05 * result["initial_loss"])
        self.assertLess(result["voltage_rmse_mV"], 1.0)
        self.assertEqual(result["saturated_parameters"], [])
        self.assertEqual(len(result["loss_history"]), 40)
        fitted = {record["key"]: record["value"] for record in result["fitted_parameters"]}
        for key, truth in HH_TRUTH.items():
            self.assertLess(abs(fitted[key] - truth) / truth, 0.15, f"{key} = {fitted[key]}, truth {truth}")

        # The exported spec re-simulates to the fitted traces and beats the start on held-out data.
        fitted_spec = common.load_json(self.root / "fit_spec" / "fitted_model.json")
        self.assertEqual(fitted_spec["name"], "hh_soma_perturbed_fitted")
        self.assertEqual(fitted_spec["trainable"], common.load_json(self.perturbed)["trainable"])
        rebuilt = common.build_model(fitted_spec, base_dir=self.root / "fit_spec")
        for key, value in fitted.items():
            self.assertAlmostEqual(float(rebuilt.nodes.iloc[0][key]), value)
        with np.load(self.root / "fit_spec" / "fitted_traces.npz") as archive:
            self.assertEqual(archive["traces"].shape, (3, 1, 2401))
        code, _ = quiet(
            evaluate_fit.main,
            [
                "--model", str(self.root / "fit_spec" / "fitted_model.json"), "--protocol", str(ASSETS / "heldout_protocol.json"),
                "--target", str(self.heldout), "--out", str(self.root / "fit_spec" / "heldout.json"),
            ],
        )
        self.assertEqual(code, 0)
        heldout = common.load_json(self.root / "fit_spec" / "heldout.json")
        self.assertLess(heldout["mean_rmse_mV"], 10.0)

    def test_random_starts_are_screened_and_the_top_k_descend_in_parallel(self) -> None:
        result = self.fit("fit_random", "--init", "random", "--n-starts", "6", "--top-k", "2", "--steps", "8", "--seed", "1")
        self.assertEqual(result["n_candidates_screened"], 6)
        self.assertEqual(result["n_candidates_descended"], 2)
        self.assertEqual(len(result["screened_losses"]), 6)
        self.assertEqual(len(result["candidate_best_losses"]), 2)
        self.assertEqual(len(result["loss_history_all_candidates"][0]), 2)
        self.assertIn(result["winner"], (0, 1))
        self.assertLessEqual(result["best_loss"], min(result["screened_losses"]))
        # The chosen initial loss is one of the two smallest screened losses.
        self.assertIn(round(result["initial_loss"], 6), [round(value, 6) for value in sorted(result["screened_losses"])[:2]])
        fitted = {record["key"]: record["value"] for record in result["fitted_parameters"]}
        for entry in common.load_json(self.perturbed)["trainable"]:
            self.assertGreaterEqual(fitted[entry["key"]], entry["lower"])
            self.assertLessEqual(fitted[entry["key"]], entry["upper"])

    def test_the_same_seed_reproduces_the_same_fit(self) -> None:
        first = self.fit("det_a", "--init", "random", "--n-starts", "4", "--top-k", "1", "--steps", "3", "--seed", "11")
        second = self.fit("det_b", "--init", "random", "--n-starts", "4", "--top-k", "1", "--steps", "3", "--seed", "11")
        self.assertEqual(first["screened_losses"], second["screened_losses"])
        self.assertEqual(first["fitted_parameters"], second["fitted_parameters"])

    def test_polyak_and_summary_loss_with_checkpointing_run_and_stay_finite(self) -> None:
        result = self.fit(
            "fit_polyak", "--optimizer", "polyak", "--loss", "summary", "--learning-rate", "0.1",
            "--steps", "5", "--checkpoint-levels", "2",
        )
        self.assertEqual(result["steps_run"], 5)
        self.assertTrue(all(np.isfinite(result["loss_history"])))
        self.assertTrue(all(np.isfinite(result["grad_norm_history"])))
        self.assertLessEqual(result["best_loss"], result["initial_loss"])

    def test_the_spike_count_loss_runs_and_reports_a_finite_history(self) -> None:
        result = self.fit("fit_spikes", "--loss", "spike_count", "--steps", "3")
        self.assertEqual(result["steps_run"], 3)
        self.assertTrue(all(np.isfinite(result["loss_history"])))

    def test_patience_stops_a_stalled_fit_early(self) -> None:
        # Starting at the truth, the loss is already ~0 and cannot improve.
        code, output = quiet(
            fit_biophysics.main,
            [
                "--model", str(ASSETS / "hh_soma.json"), "--protocol", str(ASSETS / "step_protocol.json"),
                "--target", str(self.target), "--out", str(self.root / "fit_truth"), "--steps", "50", "--patience", "3", "--log-every", "1000",
            ],
        )
        self.assertEqual(code, 0, output)
        result = common.load_json(self.root / "fit_truth" / "fit_result.json")
        self.assertIn("no improvement", result["stop_reason"])
        self.assertLess(result["steps_run"], 50)
        self.assertLess(result["voltage_rmse_mV"], 0.5)

    def test_a_target_of_the_wrong_shape_is_rejected_before_any_fitting(self) -> None:
        np.savez(self.root / "wrong.npz", traces=np.zeros((2, 1, 10)))
        code, _ = quiet(
            fit_biophysics.main,
            [
                "--model", str(self.perturbed), "--protocol", str(ASSETS / "step_protocol.json"),
                "--target", str(self.root / "wrong.npz"), "--out", str(self.root / "never"),
            ],
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "never").exists())
        code, _ = quiet(
            fit_biophysics.main,
            [
                "--model", str(self.perturbed), "--protocol", str(ASSETS / "step_protocol.json"),
                "--target", str(self.target), "--out", str(self.root / "never"), "--init", "random",
            ],
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
