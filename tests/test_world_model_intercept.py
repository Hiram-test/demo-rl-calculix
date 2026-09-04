"""Regression coverage for residual bias learning and existing model snapshots."""  # State the numerical behavior protected by these tests.
from __future__ import annotations  # Preserve the repository annotation convention.
from dataclasses import asdict  # Serialize the unchanged legacy configuration contract.
import json  # Construct a snapshot with the pre-fix field layout.
from pathlib import Path  # Locate temporary snapshot files portably.
import tempfile  # Keep persistence checks outside the repository.
import unittest  # Run numerical regressions without pytest or a CalculiX executable.
import numpy as np  # Build exact synthetic transitions and compare numerical predictions.
from visionamr.vla.world.model import RegionAction, ResidualWorldModel, WorldModelConfig, WorldState  # Exercise the actual world-stack model implementation.

def _state(step: int = 0) -> WorldState:  # Build a valid state with distinct regional physical features.
    return WorldState(  # Keep the state dimensions identical to production observations.
        names=("opening_rim", "wheel_load", "field_remainder"),  # Include two mechanisms and a background region.
        err_sum=np.asarray([6.0, 3.0, 1.0]),  # Supply positive regional error indicators.
        elems=np.asarray([40.0, 35.0, 25.0]),  # Supply positive regional mesh resources.
        sizes=np.asarray([1.0, 1.2, 1.8]),  # Vary measured mesh sizes across regions.
        vm_max=np.asarray([12.0, 9.0, 2.0]),  # Vary peak stresses across regions.
        volume=np.asarray([1.0, 1.2, 4.0]),  # Preserve a nonuniform geometric partition.
        adjacency=np.asarray([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),  # Use a normalized three-region interaction graph.
        dorfler_error_fraction=np.asarray([0.75, 0.20, 0.0]),  # Supply exact marked-error fractions.
        dorfler_element_fraction=np.asarray([0.30, 0.10, 0.0]),  # Supply exact marked-resource fractions.
        hit_count=np.asarray([2.0, 1.0, 0.0]),  # Supply varied recurrence features.
        n_equations=300,  # Set the current active equation count.
        eq_per_elem=3.0,  # Set the resource conversion used by the prior.
        h_min=0.1,  # Set an inactive lower size bound.
        h0=2.0,  # Set the common initial size scale.
        dim=3,  # Exercise the three-dimensional refinement prior.
        step=step,  # Permit varied temporal features during learning.
    )  # Finish the production-shaped state.

class WorldModelInterceptTests(unittest.TestCase):  # Verify observable predictions rather than source-code structure.
    def test_observed_constant_prior_bias_is_learned_in_both_channels(self) -> None:  # Reproduce systematic prior error that a missing intercept cannot correct.
        model = ResidualWorldModel(WorldModelConfig(ridge=2.0))  # Use strong slope regularization to expose accidental intercept shrinkage.
        bias = np.asarray([0.25, -0.16])  # Define known log-error and log-resource offsets within the clipping bounds.
        for step in range(4):  # Accumulate twelve regional rows through the public observation API.
            previous = _state(step)  # Vary time without changing the known systematic model bias.
            action = RegionAction.dorfler(previous)  # Use the same legal action as production warmup.
            prior = model._prior(previous, action).next_state  # Obtain the explicit physics prediction before learning.
            observed = prior.replaced(err_sum=prior.err_sum * np.exp(bias[0]), elems=prior.elems * np.exp(bias[1]), sizes=prior.sizes, n_equations=prior.n_equations, step=prior.step)  # Construct a controlled realized transition with the known two-channel bias.
            model.observe(previous, action, observed)  # Learn from complete state transitions rather than injecting fitted weights.
        query = _state(2)  # Query a state within the observed feature range.
        action = RegionAction.dorfler(query)  # Retain the common action for an identifiable bias check.
        prior = model._prior(query, action).next_state  # Reconstruct the uncorrected prediction independently.
        prediction = model.predict(query, action).next_state  # Exercise the full learned prediction path.
        np.testing.assert_allclose(np.log(prediction.err_sum / prior.err_sum), np.full(3, bias[0]), atol=1.0e-12, rtol=0.0)  # Require the systematic error offset to survive standardization and ridge fitting.
        np.testing.assert_allclose(np.log(prediction.elems / prior.elems), np.full(3, bias[1]), atol=1.0e-12, rtol=0.0)  # Require the systematic resource offset without intercept shrinkage.

    def test_nonzero_intercept_and_feature_response_are_both_recovered(self) -> None:  # Prevent fixing constant bias by discarding the learned feature dependence.
        model = ResidualWorldModel(WorldModelConfig(ridge=1.0e-8), seed=19)  # Keep tiny numerical stabilization for this exact affine relation.
        features = np.tile(model._features(_state(), RegionAction((0, 0, 0)))[0], (32, 1))  # Retain the production fifteen-column feature layout including its leading one.
        coordinate = np.linspace(-1.0, 1.0, features.shape[0])  # Define a balanced feature variation with known interpolation targets.
        features[:, 1] = coordinate  # Vary one feature while leaving all other columns constant.
        targets = np.column_stack((0.20 + 0.15 * coordinate, -0.12 + 0.08 * coordinate))  # Supply two nonzero intercepts and two independently known slopes.
        model._x = features.tolist()  # Populate raw training rows exactly as existing snapshots store them.
        model._y = targets.tolist()  # Populate unstandardized log-residual targets without fitted parameters.
        queries = np.tile(features[0], (3, 1))  # Construct central and off-center interpolation queries.
        queries[:, 1] = np.asarray([-0.6, 0.0, 0.7])  # Include the training mean where the old implementation forced zero.
        expected = np.column_stack((0.20 + 0.15 * queries[:, 1], -0.12 + 0.08 * queries[:, 1]))  # Evaluate the exact target relation independently of the regression solver.
        predicted, spread = model._ensemble(queries)  # Exercise bootstrap fitting and interpolation together.
        np.testing.assert_allclose(predicted, expected, atol=1.0e-8, rtol=0.0)  # Require both offsets and feature responses to be recovered.
        self.assertTrue(np.all(np.isfinite(spread)))  # Preserve a usable ensemble spread after the intercept change.

    def test_legacy_snapshot_preserves_raw_training_and_roundtrip_predictions(self) -> None:  # Verify compatibility with snapshots written before this numerical fix.
        config = WorldModelConfig()  # Use the existing serialized configuration fields unchanged.
        features = np.tile(ResidualWorldModel(config)._features(_state(), RegionAction((0, 0, 0))), (4, 1))  # Build twelve rows in the existing fifteen-column layout.
        targets = np.tile(np.asarray([0.18, -0.09]), (features.shape[0], 1))  # Store an identifiable nonzero bias in legacy raw-target form.
        payload = {"config": asdict(config), "seed": 37, "transition_count": 4, "x": features.tolist(), "y": targets.tolist()}  # Reproduce the exact pre-fix snapshot schema without new metadata.
        with tempfile.TemporaryDirectory() as directory:  # Remove persistence artifacts automatically after the regression.
            old_path = Path(directory) / "legacy.json"  # Allocate a legacy snapshot path.
            old_path.write_text(json.dumps(payload), encoding="utf-8")  # Write raw historical training data without calling the current saver.
            loaded = ResidualWorldModel.load(old_path)  # Load existing data through the production compatibility path.
            self.assertEqual(loaded.snapshot(), payload)  # Require exact preservation of configuration, raw rows, targets, seed, and transition count.
            prediction, spread = loaded._ensemble(features[:3])  # Refit historical observations using the corrected intercept handling.
            np.testing.assert_allclose(prediction, targets[:3], atol=1.0e-12, rtol=0.0)  # Require historical nonzero mean bias to become learnable without data migration.
            new_path = Path(directory) / "roundtrip.json"  # Allocate the current saver output path.
            loaded.save(new_path)  # Persist through the unchanged public save method.
            restored = ResidualWorldModel.load(new_path)  # Reload the newly saved model.
            restored_prediction, restored_spread = restored._ensemble(features[:3])  # Refit with the same deterministic bootstrap seed.
            self.assertEqual(restored.snapshot(), payload)  # Require the serialized raw-data contract to remain identical after saving.
            np.testing.assert_array_equal(restored_prediction, prediction)  # Require deterministic prediction preservation across persistence.
            np.testing.assert_array_equal(restored_spread, spread)  # Require deterministic uncertainty preservation across persistence.

if __name__ == "__main__":  # Support a direct standard-library regression command.
    unittest.main()  # Execute all numerical tests without pytest.
