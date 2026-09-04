"""One-step spatial error prediction from action-conditioned numerical features."""  # State the actual prediction scope.
from __future__ import annotations  # Keep annotations independent of import order.
import json  # Persist numerical parameters without executable pickle payloads.
from pathlib import Path  # Accept ordinary filesystem paths for model snapshots.
import numpy as np  # Provide the complete numerical backend.
# Keep training and feature construction separate so visual and scalar inputs share one estimator.
class SpatialWorldModel:  # Fit a small-sample kernel ridge model with an unpenalized intercept.
    def __init__(self, alpha: float = 10.0, model_type: str = "visual", kernel: str = "linear", gamma: float | None = None) -> None:  # Expose only fixed, caller-selected hyperparameters.
        if not np.isfinite(alpha) or alpha <= 0.0:  # Require a positive ridge term for singular small-sample problems.
            raise ValueError("alpha must be finite and positive")  # Explain the invalid numerical parameter.
        if model_type not in ("visual", "scalar"):  # Keep the input-channel label explicit in saved experiments.
            raise ValueError("model_type must be 'visual' or 'scalar'")  # Reject ambiguous experiment labels.
        if kernel not in ("linear", "rbf"):  # Restrict the implementation to the two documented estimators.
            raise ValueError("kernel must be 'linear' or 'rbf'")  # Report unsupported kernel behavior.
        if gamma is not None and (not np.isfinite(gamma) or gamma <= 0.0):  # Validate an explicitly selected RBF inverse length scale.
            raise ValueError("gamma must be finite and positive")  # Reject undefined kernel scales.
        self.alpha = float(alpha)  # Store regularization independently of the training targets.
        self.model_type = model_type  # Record the feature family without silently changing its columns.
        self.kernel = kernel  # Record the chosen kernel without tuning on evaluation data.
        self.gamma = None if gamma is None else float(gamma)  # Preserve the caller's bandwidth choice.
    @staticmethod  # Reuse strict array validation at fit and prediction boundaries.
    def _matrix(values: np.ndarray, name: str, allow_empty: bool = False) -> np.ndarray:  # Convert a numerical batch into a finite two-dimensional array.
        array = np.asarray(values, dtype=float)  # Avoid changing the caller's input arrays.
        if array.ndim != 2 or array.shape[1] == 0:  # Require explicit sample and feature axes.
            raise ValueError(f"{name} must have shape (samples, positive_width)")  # Identify the incompatible array.
        if not allow_empty and array.shape[0] == 0:  # Exclude empty training sets while allowing empty prediction batches.
            raise ValueError(f"{name} needs at least one sample")  # Explain why no estimator can be fitted.
        if not np.all(np.isfinite(array)):  # Prevent non-finite observations from contaminating linear algebra.
            raise ValueError(f"{name} contains non-finite values")  # Report the offending batch.
        return array  # Return the validated numerical view.
    def _kernel_matrix(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:  # Compute pairwise similarities in standardized training coordinates.
        if self.kernel == "linear":  # Use ordinary dual ridge as the default low-cost estimator.
            return left @ right.T  # Preserve the conventional ridge regularization scale.
        distances = np.sum(left * left, axis=1)[:, None] + np.sum(right * right, axis=1)[None, :] - 2.0 * (left @ right.T)  # Compute squared Euclidean distances without an n-by-m-by-p tensor.
        return np.exp(-self.gamma_ * np.maximum(distances, 0.0))  # Remove roundoff-negative distances before evaluating the RBF.
    def fit(self, X: np.ndarray, Y: np.ndarray, groups: np.ndarray | None = None) -> "SpatialWorldModel":  # Fit only the supplied real one-step transitions.
        x_values = self._matrix(X, "X")  # Read action-conditioned visual or scalar feature rows.
        y_values = self._matrix(Y, "Y")  # Read next-state spatial eta-squared mass divided by current total eta-squared.
        if len(x_values) != len(y_values):  # Require a target for each observed action-conditioned state.
            raise ValueError("X and Y must contain the same number of samples")  # Reject unmatched transition pairs.
        if np.any(y_values < 0.0):  # Require physically non-negative estimator masses.
            raise ValueError("Y must contain non-negative spatial error masses")  # Avoid hiding invalid targets with clipping.
        group_values = None if groups is None else np.asarray(groups)  # Accept group labels solely as training-set accounting metadata.
        if group_values is not None and (group_values.ndim != 1 or len(group_values) != len(x_values)):  # Check group alignment without choosing a split for the caller.
            raise ValueError("groups must be a one-dimensional label for every training sample")  # Explain malformed group metadata.
        self.n_samples_ = int(len(x_values))  # Record the number of real transition rows used to fit.
        self.n_features_in_ = int(x_values.shape[1])  # Fix the fitted feature schema width.
        self.n_outputs_ = int(y_values.shape[1])  # Permit 64 spatial bins and other explicitly supplied output widths.
        self.n_groups_ = None if group_values is None else int(len(np.unique(group_values)))  # Expose grouped sample accounting without retaining case identifiers.
        self.x_mean_ = np.mean(x_values, axis=0)  # Fit feature centering on training observations only.
        deviations = x_values - self.x_mean_  # Remove offsets before estimating feature variability.
        scales = np.sqrt(np.mean(deviations * deviations, axis=0))  # Measure training-only root-mean-square feature deviations.
        self.active_mask_ = scales > np.finfo(float).eps * np.maximum(1.0, np.abs(self.x_mean_))  # Identify constant and numerically constant training columns.
        self.x_scale_ = np.where(self.active_mask_, scales, 1.0)  # Avoid division by zero without creating arbitrary constant-column slopes.
        self.x_train_ = np.where(self.active_mask_, deviations / self.x_scale_, 0.0)  # Remove constant columns from every kernel computation.
        self.gamma_ = self.gamma if self.gamma is not None else 1.0 / max(int(np.sum(self.active_mask_)), 1)  # Set the default RBF bandwidth from feature dimension alone.
        transformed = np.sqrt(y_values)  # Reduce target dynamic range while retaining an exact zero representation.
        self.target_mean_ = np.mean(transformed, axis=0)  # Fit an intercept that ridge regularization never penalizes.
        centered_targets = transformed - self.target_mean_  # Separate the target offset from learned action responses.
        gram = self._kernel_matrix(self.x_train_, self.x_train_)  # Form a sample-sized matrix even when image feature width is large.
        self.kernel_mean_ = np.mean(gram, axis=0)  # Retain training kernel means for correct prediction-time centering.
        self.kernel_grand_mean_ = float(np.mean(self.kernel_mean_))  # Retain the training grand mean independently of query batches.
        centered_gram = gram - self.kernel_mean_[None, :] - self.kernel_mean_[:, None] + self.kernel_grand_mean_  # Center the feature space for both linear and nonlinear kernels.
        centered_gram = 0.5 * (centered_gram + centered_gram.T)  # Remove tiny asymmetric roundoff before solving.
        self.dual_coef_ = np.linalg.solve(centered_gram + self.alpha * np.eye(self.n_samples_), centered_targets)  # Fit all output bins with one regularized dual system.
        return self  # Support ordinary estimator chaining in the benchmark.
    def predict(self, X: np.ndarray) -> np.ndarray:  # Predict original-scale non-negative spatial masses for one action batch.
        if not hasattr(self, "dual_coef_"):  # Report use before fitting explicitly.
            raise RuntimeError("fit must be called before predict")  # Avoid silently returning an untrained prior.
        x_values = self._matrix(X, "X", allow_empty=True)  # Accept a valid empty candidate batch as a useful boundary case.
        if x_values.shape[1] != self.n_features_in_:  # Keep prediction features aligned with training features.
            raise ValueError(f"X has {x_values.shape[1]} columns; expected {self.n_features_in_}")  # Explain incompatible observation schemas.
        if len(x_values) == 0:  # Avoid undefined row means for an empty candidate set.
            return np.empty((0, self.n_outputs_), dtype=float)  # Preserve the expected two-dimensional output schema.
        standardized = np.where(self.active_mask_, (x_values - self.x_mean_) / self.x_scale_, 0.0)  # Apply training statistics and ignore columns constant during training.
        cross_gram = self._kernel_matrix(standardized, self.x_train_)  # Compare each query only with stored training observations.
        centered_cross = cross_gram - self.kernel_mean_[None, :] - np.mean(cross_gram, axis=1)[:, None] + self.kernel_grand_mean_  # Center each query against the training distribution, not its query companions.
        predicted_root = centered_cross @ self.dual_coef_ + self.target_mean_  # Restore the unpenalized transformed-target intercept.
        return np.square(np.maximum(predicted_root, 0.0))  # Invert the square-root transform without turning negative extrapolations into positive mass.
    def save(self, path: str | Path) -> Path:  # Write a transparent snapshot without training-target or evaluation-data leakage.
        if not hasattr(self, "dual_coef_"):  # Require actual fitted parameters before persistence.
            raise RuntimeError("fit must be called before save")  # Reject misleading empty snapshots.
        parameters = {"alpha": self.alpha, "model_type": self.model_type, "kernel": self.kernel, "gamma": self.gamma}  # Preserve the fixed model configuration.
        dimensions = {"n_samples": self.n_samples_, "n_features": self.n_features_in_, "n_outputs": self.n_outputs_, "n_groups": self.n_groups_}  # Record fitted observation and output dimensions.
        arrays = {name: getattr(self, name).tolist() for name in ("x_mean_", "x_scale_", "active_mask_", "x_train_", "target_mean_", "kernel_mean_", "dual_coef_")}  # Convert numerical parameters to JSON-compatible arrays.
        payload = {"schema": "visionamr-spatial-world-model-v1", "target_transform": "sqrt_nonnegative", "parameters": parameters, "dimensions": dimensions, "arrays": arrays, "gamma_fitted": self.gamma_, "kernel_grand_mean": self.kernel_grand_mean_}  # Make the one-step model representation explicit.
        destination = Path(path)  # Normalize the requested filename without changing its suffix.
        destination.parent.mkdir(parents=True, exist_ok=True)  # Create any missing output directories.
        destination.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")  # Write lossless finite numerical parameters in a compact representation.
        return destination  # Return the exact snapshot location for benchmark records.
    @classmethod  # Restore the estimator without fitting again or changing hyperparameters.
    def load(cls, path: str | Path) -> "SpatialWorldModel":  # Read and validate a previously saved snapshot.
        payload = json.loads(Path(path).read_text(encoding="utf-8"))  # Parse non-executable JSON model data.
        if payload.get("schema") != "visionamr-spatial-world-model-v1" or payload.get("target_transform") != "sqrt_nonnegative":  # Require matching prediction semantics.
            raise ValueError("unsupported spatial world-model snapshot")  # Reject incompatible snapshots clearly.
        model = cls(**payload["parameters"])  # Restore and validate the caller-selected hyperparameters.
        dimensions = payload["dimensions"]  # Read the recorded fitted matrix dimensions.
        model.n_samples_ = int(dimensions["n_samples"])  # Restore the number of kernel reference rows.
        model.n_features_in_ = int(dimensions["n_features"])  # Restore the expected feature width.
        model.n_outputs_ = int(dimensions["n_outputs"])  # Restore the spatial output width.
        model.n_groups_ = dimensions["n_groups"]  # Restore training-group accounting metadata.
        if min(model.n_samples_, model.n_features_in_, model.n_outputs_) < 1:  # Require a nonempty fitted model.
            raise ValueError("snapshot contains invalid fitted dimensions")  # Explain malformed model metadata.
        expected = {"x_mean_": (model.n_features_in_,), "x_scale_": (model.n_features_in_,), "active_mask_": (model.n_features_in_,), "x_train_": (model.n_samples_, model.n_features_in_), "target_mean_": (model.n_outputs_,), "kernel_mean_": (model.n_samples_,), "dual_coef_": (model.n_samples_, model.n_outputs_)}  # Describe every parameter's required shape.
        for name, shape in expected.items():  # Check arrays before making predictions with the restored model.
            values = np.asarray(payload["arrays"][name], dtype=float)  # Read numerical values before converting the constant-column mask.
            if values.shape != shape or not np.all(np.isfinite(values)):  # Reject corrupted parameter shapes or non-finite entries.
                raise ValueError(f"invalid snapshot array {name}")  # Identify the damaged parameter.
            if name == "active_mask_" and np.any((values != 0.0) & (values != 1.0)):  # Require a genuine binary constant-column mask.
                raise ValueError("invalid snapshot active_mask_")  # Prevent silently interpreting arbitrary numbers as booleans.
            setattr(model, name, values.astype(bool) if name == "active_mask_" else values)  # Restore the original parameter representation.
        model.gamma_ = float(payload["gamma_fitted"])  # Restore the training-only bandwidth used by the RBF kernel.
        model.kernel_grand_mean_ = float(payload["kernel_grand_mean"])  # Restore the kernel intercept correction.
        if np.any(model.x_scale_ <= 0.0) or not np.isfinite(model.gamma_) or model.gamma_ <= 0.0 or not np.isfinite(model.kernel_grand_mean_):  # Validate remaining numerical denominators and offsets.
            raise ValueError("snapshot contains invalid scaling parameters")  # Reject a snapshot that would cause invalid predictions.
        return model  # Return an estimator whose predictions match the saved model.
