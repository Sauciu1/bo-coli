import numpy as np
import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, SplineTransformer
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from scipy.interpolate import RBFInterpolator

from sklearn.metrics import mean_squared_error
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

    noise_var = np.mean((y_true - np.mean(y_true))**2)
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
            mins = arr.min(axis=0); maxs = arr.max(axis=0)
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
    result_df["dist_to_best_empirical"] = np.linalg.norm(norm.values - norm_best.values, axis=1)
    result_df.sort_values("dist_to_best_empirical", ascending=True).head(3)
    return result_df


def RbfBayesInterpolator():
    range_scaler = StandardScaler()
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, random_state=0)
    pipeline = Pipeline([("range", range_scaler), ("gpr", gpr)])
    return pipeline





class RBFRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper around scipy.interpolate.RBFInterpolator
    for scattered (x, y) -> z interpolation with optional smoothing.
    """
    def __init__(self,
                 kernel='thin_plate_spline',
                 smoothing=0.5,
                 epsilon=None,
                 degree=None):
        
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

    pipeline = Pipeline([
        ("range", range_scaler),
        ("spline", spline_transformer),
        ("ridge", ridge)
    ])
    return pipeline




def RBFKernelRegressionPipeline():
    return Pipeline([
        ("range", StandardScaler()),
        ("krr",   KernelRidge(alpha=1e-6, kernel="rbf", gamma=None))  # tune alpha & gamma
    ])



def RBFInterpolatorPipeline():
    """
    Pipeline analogous to your SplineInterpolator(), but using RBF interpolation
    with slight smoothing. Standardizes inputs, then fits an RBF surface.
    """
    range_scaler = StandardScaler()


    rbf = RBFRegressor(
        kernel='thin_plate_spline',  # smooth bending surface, great default
        smoothing=0.5,               # ~ your Ridge(alpha=0.5): slight smoothing
        epsilon=None,                # let SciPy choose (or set for Gaussian-like kernels)
        degree=None                  # no polynomial tail by default
    )

    pipeline = Pipeline([
        ("range", range_scaler),
        ("krr",   rbf),
    ])
    return pipeline



class RFKNN(BaseEstimator, RegressorMixin):
    def __init__(self, rf_params=None, n_neighbors=25):
        self.rf_params = {} if rf_params is None else rf_params
        self.n_neighbors = n_neighbors
    def fit(self, X, y):
        self.rf_ = RandomForestRegressor(**self.rf_params).fit(X, y)
        self.knn_ = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance").fit(X, self.rf_.predict(X))
        return self
    def predict(self, X):
        return self.knn_.predict(X)

def get_RfKnn_pipe():

    return Pipeline([
    ("scale", StandardScaler()),
    ("rf_smooth", RFKNN(rf_params=dict(n_estimators=400, min_samples_leaf=7), n_neighbors=30)),])


