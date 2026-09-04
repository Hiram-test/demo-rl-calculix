"""Numerical tests for one-step spatial world-model regression."""  # Describe mathematical behavior rather than benchmark outcomes.
import numpy as np  # Construct exact synthetic regression problems.
import pytest  # Parametrize kernels and inspect expected numerical errors.
from visionamr.vla.visual_world_model import SpatialWorldModel  # Import the action-conditioned numerical estimator.
# Constant targets and constant input columns expose wrongly penalized intercepts.
@pytest.mark.parametrize("kernel", ["linear", "rbf"])  # Check both feature-space centering implementations.
def test_constant_target_and_constant_columns(kernel: str) -> None:  # Verify that offsets remain exact even under strong shrinkage.
    x_values = np.column_stack([np.arange(6.0), np.ones(6), np.zeros(6)])  # Include one informative and two constant columns.
    target = np.linspace(0.0, 2.0, 64)  # Define a nonuniform constant spatial response including an exact zero.
    model = SpatialWorldModel(alpha=1.0e8, kernel=kernel).fit(x_values, np.tile(target, (6, 1)))  # Make intercept shrinkage readily detectable.
    query = np.array([[3.0, 1000.0, -999.0], [7.0, -2.0, 32.0]])  # Vary training-constant columns to check they cannot invent a learned slope.
    np.testing.assert_allclose(model.predict(query), np.tile(target, (2, 1)), atol=1.0e-14)  # Require the exact intercept in original mass units.
    np.testing.assert_array_equal(model.active_mask_, [True, False, False])  # Confirm both constant columns are excluded from the kernel.
def test_linear_ridge_matches_centered_closed_form() -> None:  # Check the dual solve against independent primal ridge with an unpenalized intercept.
    x_values = np.array([[-3.0, 4.0], [-1.0, 4.0], [0.0, 4.0], [2.0, 4.0], [5.0, 4.0]])  # Use an offset informative column and a constant column.
    roots = np.column_stack([2.0 + 0.08 * x_values[:, 0], 1.2 - 0.05 * x_values[:, 0]])  # Specify an affine response in the fitted square-root domain.
    model = SpatialWorldModel(alpha=2.5).fit(x_values, roots * roots)  # Fit without handing the estimator an explicit constant feature.
    normalized = (x_values[:, :1] - x_values[:, :1].mean(axis=0)) / x_values[:, :1].std(axis=0)  # Standardize the informative column independently.
    coefficients = np.linalg.solve(normalized.T @ normalized + 2.5 * np.eye(1), normalized.T @ (roots - roots.mean(axis=0)))  # Compute a separate centered primal ridge solution.
    expected = np.square(normalized @ coefficients + roots.mean(axis=0))  # Restore the exact transformed intercept in the reference calculation.
    np.testing.assert_allclose(model.predict(x_values), expected, rtol=1.0e-12, atol=1.0e-12)  # Detect incorrect alpha scaling or intercept handling.
@pytest.mark.parametrize("kernel", ["linear", "rbf"])  # Cover query-time centering for linear and nonlinear kernels.
def test_predictions_are_batch_independent_and_nonnegative(kernel: str) -> None:  # Verify predictions depend on each candidate and the training set only.
    x_values = np.arange(8.0).reshape(-1, 1)  # Create a one-dimensional action feature.
    y_values = np.square(0.1 + 0.2 * x_values)  # Supply monotone non-negative target masses.
    model = SpatialWorldModel(alpha=0.01, kernel=kernel).fit(x_values, y_values)  # Fit an informative synthetic transition response.
    query = np.array([[-1000.0], [2.5], [1000.0]])  # Include negative and positive extrapolation beyond the training actions.
    together = model.predict(query)  # Predict the complete candidate set at once.
    separately = np.vstack([model.predict(row.reshape(1, -1)) for row in query])  # Predict the same candidates in isolated calls.
    np.testing.assert_allclose(together, separately, rtol=1.0e-12, atol=1.0e-12)  # Reject test-batch-dependent kernel centering.
    assert np.all(np.isfinite(together)) and np.all(together >= 0.0)  # Require finite non-negative output after transformed extrapolation.
    if kernel == "linear":  # Check the direction of inverse-transform clipping where the extrapolated square root is negative.
        assert together[0, 0] == 0.0  # Ensure negative roots are clipped before squaring rather than becoming spurious mass.
@pytest.mark.parametrize("kernel", ["linear", "rbf"])  # Exercise complete persistence for both estimators.
def test_roundtrip_and_empty_predictions(kernel: str, tmp_path) -> None:  # Verify exact fitted-state serialization without storing target observations.
    rng = np.random.default_rng(210)  # Make synthetic high-dimensional inputs reproducible.
    x_values = rng.normal(size=(9, 20))  # Exercise the small-sample wide-feature regime.
    y_values = rng.uniform(size=(9, 64))  # Match the benchmark's spatial mass output dimension.
    groups = np.repeat(np.arange(3), 3)  # Account for three parent-state action groups.
    model = SpatialWorldModel(alpha=10.0, model_type="scalar", kernel=kernel, gamma=0.2).fit(x_values, y_values, groups=groups)  # Fit caller-selected hyperparameters with explicit group accounting.
    snapshot = model.save(tmp_path / "nested" / "spatial_model.json")  # Persist to a previously nonexistent output directory.
    restored = SpatialWorldModel.load(snapshot)  # Restore the fitted kernel and all training-only statistics.
    query = rng.normal(size=(4, 20))  # Generate independent prediction inputs after fitting.
    np.testing.assert_array_equal(restored.predict(query), model.predict(query))  # Require exact numerical snapshot roundtrip.
    assert restored.n_groups_ == 3 and restored.n_samples_ == 9 and restored.model_type == "scalar"  # Check fitted sample and model-type metadata.
    assert restored.predict(np.empty((0, 20))).shape == (0, 64)  # Require the expected empty-candidate output schema.
@pytest.mark.parametrize("kernel", ["linear", "rbf"])  # Ensure degenerate feature sets are supported by each kernel.
def test_single_sample_model_has_exact_intercept(kernel: str) -> None:  # Cover the smallest possible real-transition dataset.
    model = SpatialWorldModel(kernel=kernel).fit(np.array([[2.0, 3.0]]), np.array([[0.0, 0.16]]))  # Fit one observation whose feature columns are necessarily constant.
    np.testing.assert_allclose(model.predict(np.array([[200.0, -100.0]])), [[0.0, 0.16]], atol=1.0e-15)  # Return the observed intercept without inventing extrapolated slopes.
def test_invalid_targets_and_unfitted_prediction() -> None:  # Explain misuse at the regression boundary.
    with pytest.raises(RuntimeError, match="fit"):  # Check that an unfitted object cannot act as a trained world model.
        SpatialWorldModel().predict(np.ones((1, 2)))  # Attempt inference before receiving any training transitions.
    with pytest.raises(ValueError, match="non-negative"):  # Check that invalid physical labels are never silently clipped during fitting.
        SpatialWorldModel().fit(np.ones((2, 1)), np.array([[0.2], [-0.1]]))  # Supply an impossible negative spatial error mass.
    with pytest.raises(ValueError, match="groups"):  # Check action-group metadata alignment.
        SpatialWorldModel().fit(np.ones((2, 1)), np.ones((2, 1)), groups=np.array(["one"]))  # Supply one group label for two training rows.
