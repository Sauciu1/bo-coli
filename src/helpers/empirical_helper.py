import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import RBFInterpolator

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, OneToOneFeatureMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, SplineTransformer


def evaluate_model(model, X, y):
    """Calculate MSE"""

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error (MSE) of predicted vs actual: {mse:.4f}")
    print("For a total of", len(y), "data points, so RMSE is", mse**0.5)
    print(f"Which is {mse**0.5 / (y.max() - y.min()):.2%} of the range of y values")
    return mse


def total_signal_to_noise(y_true):
    """Calculate the total signal-to-noise ratio (SNR) for a vector of measurements."""
    signal_var = np.var(y_true, ddof=1)

    noise_var = np.mean((y_true - np.mean(y_true)) ** 2)
    if noise_var == 0:
        return np.inf
    return signal_var / noise_var


class ParamRangeScaler(BaseEstimator, TransformerMixin):
    """Map columns in `parameters` to [0,1] using `param_bounds` (or observed range)."""

    def __init__(self, parameters, param_bounds=None):
        self.parameters = list(parameters)
        self.param_bounds = param_bounds

    def fit(self, X, y=None):
        if self.param_bounds:
            mins = np.array([self.param_bounds[p][0] for p in self.parameters], float)
            maxs = np.array([self.param_bounds[p][1] for p in self.parameters], float)
        else:
            arr = X[self.parameters].to_numpy() if hasattr(X, "iloc") else np.asarray(X)
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
        rng = np.where(maxs - mins == 0, 1.0, (maxs - mins))
        self.mins_, self.rng_ = mins, rng
        return self

    def transform(self, X):
        arr = X[self.parameters].to_numpy() if hasattr(X, "iloc") else np.asarray(X)
        return (arr - self.mins_) / self.rng_


def norm_eucl_dist_to_best(result_df, best_empirical_obs, param_bounds):
    """Calculate the normalized Euclidean distance of each row in result_df to the best empirical observation."""
    parameters = list(param_bounds.keys())
    param_mins = pd.Series({k: v[0] for k, v in param_bounds.items()})
    param_maxs = pd.Series({k: v[1] for k, v in param_bounds.items()})
    best = best_empirical_obs.reset_index().iloc[0][parameters]
    norm = (result_df[parameters] - param_mins) / (param_maxs - param_mins)
    norm_best = (best - param_mins) / (param_maxs - param_mins)
    result_df["dist_to_best_empirical"] = np.linalg.norm(
        norm.values - norm_best.values, axis=1
    )
    result_df.sort_values("dist_to_best_empirical", ascending=True).head(3)
    return result_df


def RbfBayesInterpolator():
    range_scaler = StandardScaler()
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0
    )
    pipeline = Pipeline([("range", range_scaler), ("gpr", gpr)])
    return pipeline


class RBFRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper around scipy.interpolate.RBFInterpolator
    for scattered (x, y) -> z interpolation with optional smoothing.
    """

    def __init__(
        self, kernel="thin_plate_spline", smoothing=0.5, epsilon=None, degree=None
    ):

        self.kernel = kernel
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.degree = degree
        self._rbf = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self._rbf = RBFInterpolator(
            X,
            y,
            kernel=self.kernel,
            smoothing=self.smoothing,
            epsilon=self.epsilon,
            degree=self.degree,
        )

        return self

    def predict(self, X):
        if self._rbf is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return self._rbf(X)


def SplineInterpolator():
    range_scaler = StandardScaler()
    spline_transformer = SplineTransformer(degree=2, n_knots=7, include_bias=True)
    ridge = Ridge(alpha=0.5)

    pipeline = Pipeline(
        [("range", range_scaler), ("spline", spline_transformer), ("ridge", ridge)]
    )
    return pipeline


def RBFKernelRegressionPipeline():
    return Pipeline(
        [
            ("range", StandardScaler()),
            (
                "krr",
                KernelRidge(alpha=1e-6, kernel="rbf", gamma=None),
            ),  # tune alpha & gamma
        ]
    )


def RBFInterpolatorPipeline():
    """
    Pipeline analogous to your SplineInterpolator(), but using RBF interpolation
    with slight smoothing. Standardizes inputs, then fits an RBF surface.
    """
    range_scaler = StandardScaler()

    rbf = RBFRegressor(
        kernel="thin_plate_spline",  # smooth bending surface, great default
        smoothing=0.5,  # ~ your Ridge(alpha=0.5): slight smoothing
        epsilon=None,  # let SciPy choose (or set for Gaussian-like kernels)
        degree=None,  # no polynomial tail by default
    )

    pipeline = Pipeline(
        [
            ("range", range_scaler),
            ("krr", rbf),
        ]
    )
    return pipeline


class RFKNN(BaseEstimator, RegressorMixin):
    def __init__(self, rf_params=None, n_neighbors=25):
        self.rf_params = {} if rf_params is None else rf_params
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.rf_ = RandomForestRegressor(**self.rf_params).fit(X, y)
        self.knn_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors, weights="distance"
        ).fit(X, self.rf_.predict(X))
        return self

    def predict(self, X):
        return self.knn_.predict(X)


def plot_ci_95(
    y_matrix: np.ndarray | pd.DataFrame,
    title="Mean true_response with 95% CI across reruns",
    figure=plt.figure(figsize=(10, 5)),
    label ="Mean true_response",
):

    y_matrix = y_matrix.T
    mean = np.nanmean(y_matrix, axis=0)
    std = np.nanstd(y_matrix, axis=0)
    n = np.sum(~np.isnan(y_matrix), axis=0)
    ci95 = 1.96 * std / np.sqrt(n)

    x = np.arange(len(mean))

    mean_plot = plt.plot(x, mean, label=label)
    mean_color = mean_plot[0].get_color()
    plt.fill_between(x, mean - ci95, mean + ci95, color=mean_color, alpha=0.2, label="95% CI")
    plt.xlabel("Trial number")
    plt.ylabel("True Response")





    plt.title(title)
    plt.legend()

    return mean


def plot_bayesian_ci_quantile(
    y_matrix: np.ndarray | pd.DataFrame,
    title="Mean true_response with Bayesian 90% CI (empirical quantile) across reruns",
    figure=plt.figure(figsize=(10, 5)),
    label="Mean true_response",
    ci=0.90,
):
    """
    Plot mean and Bayesian credible interval for each trial using the empirical quantiles
    (no parametric assumption, just the quantiles of the observed values for each x).
    """
    y_matrix = y_matrix.T  # shape: (n_trials, n_repeats)
    mean = np.nanmean(y_matrix, axis=0)
    lower = np.nanquantile(y_matrix, (1 - ci) / 2, axis=0)
    upper = np.nanquantile(y_matrix, 1 - (1 - ci) / 2, axis=0)

    x = np.arange(len(mean))

    mean_plot = plt.plot(x, mean, label=label)
    mean_color = mean_plot[0].get_color()
    plt.fill_between(x, lower, upper, color=mean_color, alpha=0.2, label=f"{int(ci*100)}% Bayesian CI (quantile)")
    plt.xlabel("Trial number")
    plt.ylabel("True Response")
    plt.title(title)
    plt.legend()

    return mean


def group_trials(run, parameters, scaling_factor, threshold=0.90):
    """get mean of technical repeats. Identify true positives and false positives"""
    grouped = run.groupby(by=parameters + ["trial_name"])[
        ["pred_response", "true_response"]
    ].mean()
    grouped["true_response"] = grouped["true_response"] / scaling_factor
    grouped["pred_response"] = grouped["pred_response"] / scaling_factor

    grouped["diff"] = grouped["pred_response"] - grouped["true_response"]
    grouped.reset_index(inplace=True)
    grouped.sort_values(by="trial_name", inplace=True, ascending=True)
    grouped["trial_name"] = range(len(grouped["trial_name"]))

    grouped["pred_true"] = grouped.pred_response > threshold
    grouped["true_true"] = grouped.true_response > threshold
    grouped["true_positive"] = grouped["pred_true"] & grouped["true_true"]
    grouped["false_positive"] = grouped["pred_true"] & ~grouped["true_true"]

    return grouped


from sklearn.pipeline import Pipeline


def add_true_response(run, signal_pipeline: Pipeline, parameters):
    """Add responses that would be found given no noise"""
    run.loc[:, "true_response"] = run.apply(
        lambda row: signal_pipeline.predict(np.array([[row[p] for p in parameters]])),
        axis=1,
    )
    run = run.dropna()
    run.loc[:, "trial_name"] = (
        run["trial_name"].map(lambda x: int(x.split("_")[0])).astype(int)
    )
    run = run.sort_values(by="trial_name", ascending=True)
    return run


class ClipTransformer(OneToOneFeatureMixin, TransformerMixin, RegressorMixin):
    """Clips the input data to be within specified min and max values."""
    def __init__(self, a_min=0, a_max=1):
        self._is_fitted_ = True
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        """No fitting necessary for clipping transformer."""

        return self

    def transform(self, X):
        return np.clip(X, a_min=self.a_min, a_max=self.a_max)

    def predict(self, X):

        clip = np.clip(X, a_min=self.a_min, a_max=self.a_max)

        return clip[0][0] if len(clip) == 1 else clip
