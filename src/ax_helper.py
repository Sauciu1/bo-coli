from typing import Literal, Optional, Sequence, Tuple, Any
from webbrowser import get
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Self
from ax import Client
from ax.core import trial
from ax.api.configs import RangeParameterConfig
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.transition_criterion import MinTrials
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.adapter.registry import Generators
import logging

from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement, qMaxValueEntropy
from gpytorch.kernels import MaternKernel

from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec, ModelConfig
from ax.adapter.registry import Generators
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.transition_criterion import MinTrials
from ax.adapter.registry import Generators

from src.BayesClientManager import BayesClientManager

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float64



def silence_ax_client():
    lg = logging.getLogger("ax.api.client")
    lg.setLevel(logging.CRITICAL)      # or logging.ERROR
    lg.propagate = False
    lg.handlers.clear()                # remove any handlers Ax attached

# (Optional) also silence all other Ax loggers
def silence_all_ax():
    for name, obj in logging.root.manager.loggerDict.items():
        if name.startswith("ax"):
            lg = logging.getLogger(name)
            lg.setLevel(logging.CRITICAL)
            lg.propagate = False
            lg.handlers.clear()

silence_ax_client()


def safe_get(
    df: pd.DataFrame, key: str, index: int, axis: int = 0, default: Any = torch.nan
) -> Any:
    """Safely get a value from `df.loc[index, key]`.

    Returns `default` when the key/index is missing.
    """
    try:
        return df.loc[index, key]
    except KeyError:
        return default


def get_guess_coords(
    client: Client, output_format: Literal["df", "tensor"] = "df"
) -> Any:
    """Return guesses (arm parameters) from an Ax `client`.

    Parameters
    - client: Ax Client instance containing an experiment with trials.
    - output_format: 'df' to return a pandas.DataFrame, 'tensor' to return a
      torch.Tensor (dtype/device set by module-level variables).

    Returns
    - DataFrame or Tensor containing trial arm parameter values. The DataFrame
      uses trial index as the row index and includes a 'trial_name' column.
    """
    full_df = pd.DataFrame()
    for index, trial in client._experiment.trials.items():
        df = pd.DataFrame(
            {"trial_name": trial.arm.name, **trial.arm.parameters}, index=[index]
        )
        full_df = pd.concat([full_df, df], ignore_index=True)
    if output_format.lower() == "df":
        return full_df
    elif output_format.lower() == "tensor":
        # exclude string-like columns (e.g., trial_name) when building tensor
        numeric_cols = [c for c in full_df.columns if c != "trial_name"]
        return torch.as_tensor(full_df[numeric_cols].values, dtype=dtype, device=device)


def get_obs_from_client(client: Client) -> pd.DataFrame:
    """Fetch existing trial parameter values and attach observed responses.

    The returned DataFrame contains the trial parameters (one row per trial)
    and a column named `response_col` containing the observed mean (or NaN
    if not available).
    """
    obs = get_guess_coords(client, output_format="df")
    results = client._experiment.fetch_data().df
    response_col = list(client._experiment.metrics.keys())[0]

    def get_ax_mean(result):
        if result["trial_index"] ==80:
            print(result)
        if "mean" in result:
            return result["mean"]
        else:
            return torch.nan

    obs[response_col] = np.empty_like(obs.index, dtype=float)

    obs[response_col] = results.reindex(obs.index)["mean"]
    
    return obs



    


def get_train_Xy(
    trial_df: pd.DataFrame, dim_cols: Sequence[str], response_col: str = "response"
) -> Tuple[Tensor, Tensor]:
    """Extract training tensors X and y from an observation DataFrame.

    Returns
    - train_X: shape (n_dims, n_points) tensor of inputs (dtype/device from module)
    - train_Y: shape (n_points, 1) tensor of outputs
    """
    if response_col not in trial_df.columns:
        raise KeyError(f"Response column '{response_col}' not found in DataFrame.")
    trial_df = trial_df.dropna(subset=[response_col])
    X_vals = trial_df[list(dim_cols)].values
    y_vals = trial_df[response_col].values
    # Use shape (n_points, n_dims) which is the common convention for models
    train_X = torch.as_tensor(X_vals, dtype=dtype, device=device)
    train_Y = torch.as_tensor(y_vals, dtype=dtype, device=device).unsqueeze(-1)
    return train_X, train_Y



def ax_param_bounds_as_list(parameters:Sequence[RangeParameterConfig]):
    """returns parameter bounds in simpler to use format"""

    return np.array([p.bounds for p in parameters])

class UnitCubeScaler(BaseEstimator, TransformerMixin):
    """Scale parameters to/from the unit hypercube.

    Accepts an optional iterable of Ax `RangeParameterConfig`-like objects
    at construction. If provided, their `.bounds` are used during `fit` and
    `transform`.
    """

    def __init__(
        self, ax_parameters: Optional[Sequence[RangeParameterConfig]] = None
    ) -> None:
        # store provided parameters for later use in fit
        self.parameters = ax_parameters
        self.bounds: Optional[np.ndarray] = (
            ax_param_bounds_as_list(ax_parameters) if ax_parameters is not None else None
        )
        self._output_type = "default"
        self.dim_names: Optional[Sequence[str]] = None



    def set_output(self, *, transform: Optional[str] = None):  # sklearn >=1.2 style
        """Set output container type.

        Parameters
        ----------
        transform : {'pandas', 'default', None}
            If 'pandas', subsequent transform calls return a pandas DataFrame.
            If 'default' or None, returns a numpy ndarray.
        """
        if transform == "pandas":
            self._output_type = "pandas"
        elif transform in {None, "default"}:
            self._output_type = "default"
        else:
            raise ValueError("transform must be one of {'pandas','default',None}")
        return self

    def fit(self, X, y=None):
        # y is unused, but kept for compatibility with sklearn
        if isinstance(X, pd.DataFrame):
            self.dim_names = list(X.columns)
        else:
            self.dim_names = None
        # If parameters were provided at construction, prefer those bounds
        if getattr(self, "parameters", None) is not None:
            # parameters may be an iterable of RangeParameterConfig or similar
            try:
                self.bounds = np.array([p.bounds for p in self.parameters])
                # also set dim_names from parameter objects if they expose a name
                try:
                    self.dim_names = [getattr(p, "name", None) for p in self.parameters]
                except Exception:
                    pass
            except Exception:
                # fallback to computing bounds from X
                X = np.asarray(X)
                self.bounds = np.column_stack((np.min(X, axis=0), np.max(X, axis=0)))
        else:
            X = np.asarray(X)
            self.bounds = np.column_stack((np.min(X, axis=0), np.max(X, axis=0)))
        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        X_arr = X.values if is_df else np.asarray(X)
        mins = self.bounds[:, 0]
        maxs = self.bounds[:, 1]
        # guard against zero-width bounds
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        X_scaled = (X_arr - mins) / ranges
        if self._output_type == "pandas":
            # Column names preference: self.dim_names (if set) else passed DataFrame columns else generic names
            if self.dim_names is not None:
                columns = list(self.dim_names)
            elif is_df:
                columns = list(X.columns)
            else:
                columns = [f"x{i}" for i in range(X_scaled.shape[1])]
            index = X.index if is_df else None
            return pd.DataFrame(X_scaled, index=index, columns=columns)
        return X_scaled

    def inverse_transform(self, X_scaled):
        is_df = isinstance(X_scaled, pd.DataFrame)
        X_arr = X_scaled.values if is_df else np.asarray(X_scaled)
        mins = self.bounds[:, 0]
        maxs = self.bounds[:, 1]
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        X_orig = X_arr * ranges + mins
        if self._output_type == "pandas":
            if self.dim_names is not None:
                columns = list(self.dim_names)
            elif is_df:
                columns = list(X_scaled.columns)
            else:
                columns = [f"x{i}" for i in range(X_orig.shape[1])]
            return pd.DataFrame(
                X_orig, index=X_scaled.index if is_df else None, columns=columns
            )
        return X_orig

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    # sklearn feature name interface
    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None):
        """Return feature names in the correct order for pandas output.

        If the scaler has learned `dim_names`, use them; otherwise base on
        provided `input_features` or generate generic names.
        """
        if self.dim_names is not None:
            return np.array(list(self.dim_names), dtype=object)
        if input_features is not None:
            return np.array(list(input_features), dtype=object)
        # fallback generic naming
        if self.bounds is not None:
            n = self.bounds.shape[0]
        else:
            n = 0
        return np.array([f"x{i}" for i in range(n)], dtype=object)




class BatchClientHandler:
    def __init__(
        self,
        client: Client,
        response_function: callable,
        dim_names: str,
        response_col: str = "response",
        batch_size=8,
        range_params: Optional[Sequence[RangeParameterConfig]] = None,
    ):
        self.client: Client = client
        self.response_function: callable = response_function
        self.dim_names: list[str] = dim_names
        self.response_col: str = response_col
        self.batch_size: int = batch_size
        self.range_params: Optional[Sequence[RangeParameterConfig]] = range_params

    def get_next_batch(self, batch_size: int = None):
        """Request the next batch of trials from the Ax client."""
        if batch_size is None:
            batch_size = self.batch_size

        self.client.get_next_trials(max_trials=batch_size)

    def complete_all_pending(self):
        """Complete all pending trials by evaluating the response function."""
        for i, trial in get_obs_from_client(
            self.client).iterrows():
            if not pd.isna(trial[self.response_col]):
                continue

            response = self.response_function(**trial[self.dim_names])
            self.client.complete_trial(
                trial_index=i, raw_data={self.response_col: float(response)}
            )

    def get_pending_trials(self):
        """Return a DataFrame of pending trials (no observed response yet)."""
        obs_df = self.get_batch_observations()
        pending_mask = pd.isna(obs_df[self.response_col])
        return obs_df.loc[pending_mask]

    def get_batch_observations(self):
        """Return a DataFrame of all trials with observed responses."""
        obs =  get_obs_from_client(self.client)
        obs['trial_index'] = obs['trial_name'].apply(lambda x: x.split('_')[0]).astype(int)
        #del obs['trial_name']
        return obs

    def plot_GP(self, gp: callable, coords=None, **kwargs):
        from src.GPVisualiser import GPVisualiserPlotly

        obs = self.get_batch_observations()
        plotter = GPVisualiserPlotly.init_from_client(self.client, gp)
 
        if coords is None:
            coords = obs.loc[obs[self.response_col].idxmax(), self.dim_names].tolist()

        plotter.plot_all(coords, **kwargs, render = False)
        return plotter.fig

    def comp_noise_and_repeats(
        self,
        noise_fn: callable = None,
        repeats: int = 1,
    ):
        """Apply noise function to all running trials, and repeat each trial a specified number of times."""

        for index, trial in list(self.client._experiment.trials.items()):

            if not trial.status.is_running:
                continue
            params = trial.arms[0].parameters

            def response(coords):
                return float(noise_fn(x=coords, y=self.response_function(**coords)))

            self.client.complete_trial(
                index, raw_data={self.response_col: response(params)}
            )

            for i in range(repeats - 1):
                self.client.attach_trial(
                    parameters=params,
                    arm_name=trial.arm.name,
                )

                self.client.complete_trial(
                    trial_index=len(self.client._experiment.trials) - 1,
                    raw_data={self.response_col: response(params)},
                )


    def get_grouped_obs(self):
        df = self.get_batch_observations().dropna(subset=[self.response_col]).copy()
        params = list(self.dim_names)
        keys = df[params].apply(tuple, axis=1)
        df["group"] = pd.factorize(keys)[0] + 1
        return df
    
    def get_group_stats(self):

        grouped = self.get_grouped_obs().groupby('group')
        # compute std of predicted response per group (Series indexed by group)
        pred_response_var = grouped[self.response_col].std()

        # compute means only over numeric columns to avoid TypeError on object dtypes
        means = grouped.mean(numeric_only=True)

        pred_response_var = pred_response_var.reindex(means.index)

        if "trial_index" in means.columns:
            means = means.drop(columns=["trial_index"])

        means["pred_response_var"] = pred_response_var.values
        means.dropna(inplace=True)
        return means
        

from botorch.acquisition import qLogExpectedImprovement
from botorch.models import SingleTaskGP

class SequentialRuns:
    """Simulate sequential Bayesian optimization with batches and technical repeats.

    Initialized with a test (objective) function so different objectives can reuse the logic.
    """

    def __init__(self, test_fn, range_parameters, dim_names, metric_name):
        self.test_fn = test_fn
        self.range_parameters = range_parameters
        self.dim_names = dim_names
        self.metric_name = metric_name

    def run(
        self,
        gp=SingleTaskGP,
        acqf_class=qLogExpectedImprovement,
        n_runs=10,
        batch_size=1,
        technical_repeats=1,
        noise_fn=None,
        plot_each=False,
    ):

        client = None

        def _bounds_dict_to_range_params(bounds: dict):
            params = []
            for name, cfg in bounds.items():
                log = cfg.get("log_scale", False)
                params.append(
                    RangeParameterConfig(
                        name=name,
                        parameter_type="float",
                        bounds=(cfg["lower_bound"], cfg["upper_bound"]),
                        scaling="log" if log else "linear",
                    )
                )
            return params


        if hasattr(self.range_parameters, "_create_ax_client") and callable(
            getattr(self.range_parameters, "_create_ax_client")
        ):
            manager = self.range_parameters
            client = manager._create_ax_client()
        else:
            # If a bounds dict was passed
            if isinstance(self.range_parameters, dict):
                params = _bounds_dict_to_range_params(self.range_parameters)
            else:
                # assume it's already a sequence of RangeParameterConfig
                params = self.range_parameters

            client = Client()
            client.configure_experiment(parameters=params)
            client.configure_optimization(objective=self.metric_name)

        if noise_fn is None:
            def noise_fn(x):
                return x

        generation_strategy = get_full_strategy(gp=gp, acqf_class=acqf_class)
        client.set_generation_strategy(generation_strategy=generation_strategy)

        self.handler = BatchClientHandler(
            client,
            self.test_fn,
            self.dim_names,
            self.metric_name,
            batch_size=batch_size,
            range_params=self.range_parameters,
        )
        self.handler.get_next_batch(batch_size)

        for _ in range(n_runs):
            self.handler.comp_noise_and_repeats(noise_fn=noise_fn, repeats=technical_repeats)
            self.handler.get_next_batch()
            if plot_each:
                self.handler.plot_GP(gp, figsize=(8, 4))
                #plot_test()
                #plt.show()
        return self.handler
    







def construct_simple_surrogate(gp: callable, kernel=None):
    if kernel is None:
        return SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=SingleTaskGP,
                ),
            ]
        )
    else:
        raise NotImplementedError("Custom kernel not implemented yet.")


def construct_gen_spec(
    surrogate_spec: SurrogateSpec, acqf_class: callable, acqf_options: dict = None
):
    generator_spec = GeneratorSpec(
        generator_enum=Generators.BOTORCH_MODULAR,
        model_kwargs={
            "surrogate_spec": surrogate_spec,
            "botorch_acqf_class": qLogExpectedImprovement,
            # Can be used for additional inputs that are not constructed
            # by default in Ax. We will demonstrate below.
            "acquisition_options": {},
        },
        # We can specify various options for the optimizer here.
        model_gen_kwargs={
            "model_gen_options": {
                "optimizer_kwargs": {
                    "num_restarts": 20,
                    "sequential": False,
                    "options": {
                        "batch_limit": 3,
                        "maxiter": 200,
                    },
                },
            },
        },
    )
    return generator_spec



def construct_generation_strategy(
    generator_spec: GeneratorSpec, node_name: str, transition_trials: int = 5
) -> GenerationStrategy:
    """Constructs a Center + Sobol + Modular BoTorch `GenerationStrategy`
    using the provided `generator_spec` for the Modular BoTorch node.
    """
    botorch_node = GenerationNode(
        node_name=node_name,
        generator_specs=[generator_spec],
    )

    # Sobol for initial space exploration
    sobol_node = GenerationNode(
        node_name="Sobol",
        generator_specs=[
            GeneratorSpec(
                generator_enum=Generators.SOBOL,
            ),
        ],
        transition_criteria=[
            # Transition to BoTorch node once there are `transition_trials` trials on the experiment.
            MinTrials(
                threshold=transition_trials,
                transition_to=botorch_node.node_name,
                use_all_trials_in_exp=True,
            )
        ],
    )
    # Center node is a customized node that uses a simplified logic and has a
    # built-in transition criteria that transitions after generating once.
    center_node = CenterGenerationNode(next_node_name=sobol_node.node_name)
    return GenerationStrategy(
        name=f"Center+Sobol+{node_name}", nodes=[center_node, sobol_node, botorch_node]
    )



def get_full_strategy(gp: callable, acqf_class: callable, kernel=None, transition_trials: int = 5):
    surrogate_spec = construct_simple_surrogate(gp=gp, kernel=kernel)
    generator_spec = construct_gen_spec(
        surrogate_spec=surrogate_spec,
        acqf_class=acqf_class,
    )
    generation_strategy = construct_generation_strategy(
        generator_spec=generator_spec,
        node_name=f"{gp.__name__}+{acqf_class.__name__}",
        transition_trials=transition_trials
    )


    return generation_strategy




def get_y_data(runs, dim_names, test_func):

    df = runs.get_batch_observations().sort_values(by='trial_index', ascending=True)
    df = df.groupby(["trial_index", *dim_names])['response'].mean().reset_index()

    df['y_true'] = df[dim_names].apply(lambda r: float(test_func(*r.values)), axis=1)
    df['error'] = df['response'] - df['y_true']
    return df


def get_above_percentile(df, max_val, percentile = 0.95):
    cut_off = percentile*max_val
    #df = df[(df['response']>cut_off) & (df['y_true']>cut_off)]
    df["assumed_hit"] = df['response']>cut_off
    df["true_hit"] = df['y_true']>cut_off


    return df


if __name__ == "__main__":
    range_parameters = [
    RangeParameterConfig(name="x1", parameter_type="float", bounds=(-2, 10))
]

metric_name = "response"
dim_names = ["x1"]


def test_function(x1):
    # Original: y = -((x1 - 2) ** 2) + 10  -> range on [-2,10] is [-54, 10]
    y = -((x1 - 2) ** 2) + 10
    return (y + 54) / 64  # scale to [0,1]


noise_fn = lambda x, y: y + np.random.normal(0, 1)


def plot_test(fig=None):
    range_parameters = [
        RangeParameterConfig(name="x1", parameter_type="float", bounds=(-2, 10))
    ]

    metric_name = "response"
    dim_names = ["x1"]


    def test_function(x1):
        # Original: y = -((x1 - 2) ** 2) + 10  -> range on [-2,10] is [-54, 10]
        y = -((x1 - 2) ** 2) + 10
        return (y + 54) / 64  # scale to [0,1]


    noise_fn = lambda x, y: y + np.random.normal(0, 1)


    tester = SequentialRuns(test_function, range_parameters, dim_names, metric_name)

    runs = tester.run(
        SingleTaskGP,
        n_runs=4,
        technical_repeats=1,
        batch_size=3,
        noise_fn=noise_fn,
        plot_each=False,
        )
    
    
    data = runs.get_batch_observations()

    return data


if __name__ == "__main__":
    print(plot_test())



