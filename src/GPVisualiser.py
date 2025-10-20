# Created by Povilas (GitHub: Sauciu1) on 2025-09-15
# last updated on 2025-10-05 by Povilas - Refactored for better integration with BayesPlotter


from ax import Client
import pandas as pd
import torch
from torch import Tensor
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer
import tempfile, webbrowser, plotly.io as pio
from src.ax_helper import UnitCubeScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.BayesClientManager import BayesClientManager


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Default dtype for the visualiser tensors - keep high precision by default
dtype = torch.float64

def subplot_dims(n) -> tuple[int, int]:
    """Get grid dimensions for plotting based on number of results."""
    length = math.floor(n**0.5)
    return length, math.ceil(n / length)


class GPVisualiser:
    def __init__(self, bayes_manager: BayesClientManager) -> None:
        if hasattr(bayes_manager, 'sync_self'):
            bayes_manager.sync_self()
        self.bayes_manager = bayes_manager
        self.response_label = bayes_manager.response_label
        self.feature_labels = bayes_manager.feature_labels
        self.group_label = bayes_manager.group_label
        
        # Setup scaler
        feature_range_params = bayes_manager._ax_parameters
        if feature_range_params is not None:
            self.scaler = UnitCubeScaler(ax_parameters=feature_range_params)
            self.scaler.set_output(transform="pandas")
        else:
            self.scaler = FunctionTransformer(lambda x: x, validate=False)

        # Train GP model
        self.gp = self._train_gp(self.bayes_manager.gp)
        self.subplot_dims = subplot_dims(self.obs_X.shape[1])
        self.fig = None

    @property
    def get_obs_X_y(self):
        """Get observed and predicted data separated by NA values."""
        obs = self.bayes_manager.data
        mask_na = obs[[self.response_label] + self.feature_labels].isna().any(axis=1)

        self.predict_X = obs.loc[mask_na, self.feature_labels]
        self.predict_y = obs.loc[mask_na, self.response_label]
        self.predict_groups = obs.loc[mask_na, self.group_label] if self.group_label in obs.columns else None
        
        self.obs_X = obs.loc[~mask_na, self.feature_labels]
        self.obs_y = obs.loc[~mask_na, self.response_label]
        self.obs_groups = obs.loc[~mask_na, self.group_label] if self.group_label in obs.columns else None
        return self.obs_X, self.obs_y
    
    @property
    def obs_X_vals(self):
        return self.get_obs_X_y[0].values
    
    @property
    def obs_y_vals(self):
        return self.get_obs_X_y[1].values


    def _train_gp(self, gp: callable):
        train_X = self.scaler.fit_transform(self.obs_X_vals)

        if isinstance(train_X, pd.DataFrame):
            train_X = train_X.values

        train_X = torch.tensor(train_X, dtype=dtype, device=device)
        train_Y = torch.tensor(self.obs_y_vals, dtype=dtype, device=device).unsqueeze(
            -1
        )

        return gp(train_X, train_Y)
        

    def _create_linspace(self, num_points: int = 300) -> list[Tensor]:
        linspaces = []
        self.bayes_manager.bounds
        for name, bound in self.bayes_manager.bounds.items():
            grid = torch.linspace(
                bound['lower_bound'] * 0.95,
                bound['upper_bound'] * 1.05,
                num_points,
                dtype=dtype,
                device=device,
            )
            linspaces.append(grid)
        return linspaces
    
    def _eval_gp(self, test_X:Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate the GP model at given test points."""

        test_X = pd.DataFrame(self.scaler.transform(test_X))


        test_X = torch.tensor(test_X.values, dtype=dtype, device=device)

        with torch.no_grad():
            posterior = self.gp.posterior(test_X)
            mean = posterior.mean.squeeze()
            std = posterior.variance.sqrt().squeeze()
        return mean, std

    def _get_plane_gaussian_for_point(
        self, coordinates: Tensor, fixed_dim: int, grid: Tensor
    ) -> tuple:
        """returns mean and std of GP prediction along a grid in fixed_dim, holding other dims at coordinates"""

        zeros = torch.zeros_like(grid)
        test_X = torch.stack(
            [
                zeros + coordinates[i] if i != fixed_dim else grid
                for i in range(len(coordinates))
            ],
            dim=-1,
        )

        mean, std = self._eval_gp(test_X)

        return mean, std

    @staticmethod
    def _get_euclidean_distance(
        point1: list[float] | Tensor, point2: list[float] | Tensor
    ) -> float:
        """Calculate Euclidean distance between two points."""
        if not isinstance(point1, Tensor):
            point1 = torch.tensor(point1, dtype=dtype)
        if not isinstance(point2, Tensor):
            point2 = torch.tensor(point2, dtype=dtype)
        return torch.dist(point1, point2).item()
    
    def get_best_observed_coord(self) -> pd.Series:
        """Get the coordinates of the best observed point using BayesClientManager."""
        best_coords = self.bayes_manager.get_best_coordinates()
        if best_coords is not None:
            return pd.Series(best_coords)
        # Fallback to local calculation
        best_idx = self.obs_y.idxmax()
        return self.obs_X.loc[best_idx]

    def plot_all(
        self, coordinates: list[float] | Tensor | pd.Series, linspace=None, figsize=(12,6)
    ) -> None:
        """Handle plotting all dimensions. One subplot per dimension.
        Requires definition of:
        * self.create_subplot,
        * self.plot_gp,
        * self._plot_observations,
        * self._vlines,
        * self._add_subplot_elements
        """

        if coordinates is None:
            coordinates = self.get_best_observed_coord()
            

        # Normalize coordinates into a torch.Tensor (avoid torch.tensor(tensor))
        if isinstance(coordinates, pd.Series):
            coordinates = torch.tensor(coordinates, dtype=dtype)
        elif isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=dtype)


        self.fig, axs = self._create_subplots(figsize=figsize)

        linspace = self._create_linspace()

        for i in range(self.obs_X.shape[1]):
            ax: plt.Axes = axs[i]

            if not isinstance(linspace[i], Tensor):
                grid = torch.tensor(
                    linspace[i], dtype=dtype, device=device
                )
            else:
                grid = linspace[i].detach().clone()


            mean, std = self._get_plane_gaussian_for_point(
                coordinates, fixed_dim=i, grid=grid
            )

            self._plot_gp(grid, mean, std, ax, coordinates)
            dim = self.obs_X.columns[i]

            self._vlines(ax, coordinates[i])

            self._plot_observations(ax, dim, coordinates)

            self._add_expected_improvement(ax, dim, coordinates)

        rounded_coords = [f"{x:.3g}" for x in coordinates]
        self._add_subplot_elements(rounded_coords)

        return self.fig, axs
    
    def _get_distance_to_plane(self, obs:pd.DataFrame, coord:Tensor, fixed_dim:int) -> Tensor:
        """get distance from each observation to the plane parallel to fixed_dim and passing through coord"""

        if not isinstance(obs, Tensor):
            obs = torch.tensor(obs.values, dtype=dtype)
        if not isinstance(coord, Tensor):
            coord = torch.tensor(coord, dtype=dtype)

        obs[:, fixed_dim] = coord[fixed_dim]


        dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(obs, coord), 2), dim=1)) 
        return dist.abs()



    def _get_size(self, obs:pd.DataFrame, coordinates:Tensor, dim) -> list[float]:
        if not isinstance(dim, int):
            dim = self.obs_X.columns.get_loc(dim)

        distances =  self._get_distance_to_plane(obs, coordinates, dim)

        return torch.tensor([(1 - d / max(distances+torch.tensor([0.001], dtype=dtype))) for d in distances], dtype=dtype)
    

    def _add_expected_improvement(self, ax:plt.Axes, dim:str, coordinates:Tensor):
        
        mean, std = self._eval_gp(self.predict_X)
        
        dim_x = self.predict_X.loc[:, dim]

        sizes = self._get_size(self.predict_X, coordinates, dim)

        self._plot_expected_improvement(ax, dim_x, mean, std, sizes)

    def _plot_expected_improvement():
        raise NotImplementedError
    
    @classmethod
    def init_from_client(cls, client: Client, gp =None):
        """Initialize GPVisualiser from raw data and a GP constructor. Legacy hook for backwards compatibility."""
        from botorch.models import SingleTaskGP

        bayes_manager = BayesClientManager.init_from_client(client)
        if gp is not None:
            bayes_manager._gp = gp
        return cls(bayes_manager=bayes_manager)



class GPVisualiserMatplotlib(GPVisualiser):
    def _plot_expected_improvement(self, ax, x, mean, std, sizes):
        if len(x) == 0:
            return
        elif len(x) == 1:
            x, mean, std, sizes = [x], [mean], [std], [sizes]

        # Get group labels for predicted points if available
        group_labels = self.predict_groups.values if self.predict_groups is not None else [None] * len(x)

        # Plot each predicted point as a separate error bar with group info
        for xi, mi, si, sz, group in zip(x, mean, std, sizes, group_labels):
            group_text = f"G{int(group)}" if group is not None else "Unknown"
            ax.errorbar(
                xi,
                mi,
                yerr=2 * si,
                fmt='',
                color='red',
                alpha=0.3,
                linewidth=sz * 1 + 0.1,
                capsize=sz * 1 + 0.1,
                label='Predicted observation',
                picker=True,
                gid=f"Group: {group_text} | x={float(xi):.3g}, y={float(mi):.3g}"
            )


    @staticmethod
    def _plot_gp(grid: Tensor, mean: Tensor, std: Tensor, ax: plt.Axes, coordinates):
        """Plot GP mean and confidence intervals along a single dimension."""

        ax.plot(grid.numpy(), mean, label="GP mean")
        ax.fill_between(
            grid.numpy(), mean - 2 * std, mean + 2 * std, alpha=0.3, label="95% CI"
        )


    def _vlines(self, ax, coordinates):
        bounds = ax.get_ylim()
        ax.vlines(
            x=coordinates,
            ymin=bounds[0],
            ymax=bounds[1],
            color="k",
            linestyles="--",
            label="Current Dim Value",
            alpha=0.7,
        )



    def _create_subplots(self, figsize=(12,6)):
        fig, axs = plt.subplots(*self.subplot_dims, figsize=figsize)
        axs = axs.flatten() if self.obs_X.shape[1] > 1 else [axs]
        return fig, axs

    def _add_subplot_elements(self, rounded_coords):
        # Use the Figure's suptitle so we don't rely on pyplot state
        # Center and bold the suptitle
        self.fig.suptitle(
            f"GP Along Each Dimension for point {rounded_coords}",
            fontsize=20,
            fontweight='bold',
            x=0.5,
            ha='center'
        )

        handles, labels = self.fig.axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        self.fig.legend(
            unique.values(),
            unique.keys(),
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
        )

        self.fig.tight_layout(rect=[0, 0, 0.86, 0.95])

    def _plot_observations(
        self,
        ax: plt.Axes,
        dim_name: str,
        coordinates,
    ) -> None:
        """Plot observed data points along a single dimension with group information."""
        dim = self.obs_X.columns.get_loc(dim_name)
        point_size = self._get_size(self.obs_X, coordinates, dim)

        # Create scatter plot with default colors (no group-based coloring)
        ax.scatter(
            self.obs_X[dim_name],
            self.obs_y,
            s=point_size*100+1,
            edgecolor="k",
            alpha=0.7,
            label="Observations"
        )

        ax.set_title(f"GP along {dim_name}")
        ax.legend().set_visible(False)

    def return_fig(self):
        return self.fig
    


class PlotlyAxWrapper:
    """Wrapper to make Plotly subplots behave like matplotlib axes."""
    def __init__(self, fig, row, col):
        self.fig = fig
        self.row = row
        self.col = col

class GPVisualiserPlotly(GPVisualiser):
    def _plot_expected_improvement(self, ax, x, mean, std, sizes):
        if len(x) == 0:
            return
        elif len(x) == 1:
            x, mean, std, sizes = [x.item()], [mean.item()], [std.item()], [sizes.item()]
        else:
            # Convert tensors to numpy/python types
            x = x.cpu().numpy() if hasattr(x, 'cpu') else x.values
            mean = mean.cpu().numpy() if hasattr(mean, 'cpu') else mean
            std = std.cpu().numpy() if hasattr(std, 'cpu') else std
            sizes = sizes.cpu().numpy() if hasattr(sizes, 'cpu') else sizes

        # Get group labels for predicted points
        group_labels = self.predict_groups.values if self.predict_groups is not None else [None] * len(x)

        # Plot each predicted point as a separate error bar with group info
        for xi, mi, si, sz, group in zip(x, mean, std, sizes, group_labels):
            group_text = f"G{int(group)}" if group is not None else "Unknown"
            hover_text = f"Predicted Group {group_text}<br>x: {float(xi):.3g}<br>y: {float(mi):.3g}±{float(2*si):.3g}"
            
            # Error bar line
            ax.fig.add_trace(
                go.Scatter(
                    x=[float(xi), float(xi)],
                    y=[float(mi - 2 * si), float(mi + 2 * si)],
                    mode='lines',
                    line=dict(color='#D45AFF', 
                              width=float(sz * 10 + 5)),
                    opacity=0.3,
                    showlegend=False,
                    name='Predicted (selected point)',
                    hovertext=hover_text,
                    hoverinfo='text'
                ),
                row=ax.row, col=ax.col
            )
            


    @staticmethod
    def _plot_gp(grid: Tensor, mean: Tensor, std: Tensor, ax, coordinates):
        """Plot GP mean and confidence intervals along a single dimension."""
        grid_np = grid.cpu().numpy()
        mean_np = mean.cpu().numpy()
        std_np = std.cpu().numpy()
        
        # Add confidence interval
        ax.fig.add_trace(
            go.Scatter(
                x=np.concatenate([grid_np, grid_np[::-1]]),
                y=np.concatenate([mean_np - 2 * std_np, (mean_np + 2 * std_np)[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=False
            ),
            row=ax.row, col=ax.col
        )
        
        # Add mean line
        ax.fig.add_trace(
            go.Scatter(
                x=grid_np,
                y=mean_np,
                mode='lines',
                name='GP mean',
                line=dict(color='blue'),
                showlegend=False
            ),
            row=ax.row, col=ax.col
        )

    def _vlines(self, ax, coordinates):
        # Get y-axis range for the subplot
        y_range = [float(self.obs_y.min() * 0.99), float(self.obs_y.max() * 1.01)]
        
        ax.fig.add_trace(
            go.Scatter(
                x=[float(coordinates), float(coordinates)],
                y=y_range,
                mode='lines',
                line=dict(color='black', dash='dash', width=2),
                opacity=0.7,
                name='Current Dim Value',
                showlegend=False
            ),
            row=ax.row, col=ax.col
        )

    def _create_subplots(self, figsize=(12, 6)):
        rows, cols = self.subplot_dims
        
        # Create subplot titles
        subplot_titles = [f"GP along {col}" for col in self.obs_X.columns]
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )


        width_px = int(figsize[0] * 100)
        height_px = int(figsize[1] * 100)


        fig.update_layout(width=width_px, height=height_px)

        # Create wrapper objects for each subplot
        axs = []
        for i in range(self.obs_X.shape[1]):
            row = (i // cols) + 1
            col = (i % cols) + 1
            axs.append(PlotlyAxWrapper(fig, row, col))

        return fig, axs

    def _add_subplot_elements(self, rounded_coords):
        # Update layout
        # Only set title and legend here; sizing is handled in _create_subplots
        self.fig.update_layout(
            title=dict(
                text=f"GP Along Each Dimension for point {rounded_coords}",
                x=0.5,
                xanchor='center',
            ),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=0.5,
                xanchor="left",
                yanchor="middle",
            ),
        )

        # Add legend by making first occurrence of each trace name visible
        seen_names = set()
        for trace in self.fig.data:
            if trace.name and trace.name not in seen_names:
                trace.showlegend = True
                seen_names.add(trace.name)

    def _plot_observations(self, ax, dim_name: str, coordinates):
        """Plot observed data points along a single dimension with group information."""
        dim = self.obs_X.columns.get_loc(dim_name)
        point_size = self._get_size(self.obs_X, coordinates, dim)
        
        # Convert tensors to numpy for plotly
        if hasattr(point_size, 'cpu'):
            point_size = point_size.cpu().numpy()
        elif hasattr(point_size, 'numpy'):
            point_size = point_size.numpy()
        
        # Prepare hover text with group information but use default colors
        if self.obs_groups is not None:
            hover_text = [f"Group {int(g)}<br>x: {x:.3g}<br>y: {y:.3g}" 
                         for x, y, g in zip(self.obs_X[dim_name].values, 
                                          self.obs_y.values, 
                                          self.obs_groups.values)]
        else:
            hover_text = [f"x: {x:.3g}<br>y: {y:.3g}" 
                         for x, y in zip(self.obs_X[dim_name].values, self.obs_y.values)]
        
        # Use default orange color for all observations
        colors = 'orange'
        
        ax.fig.add_trace(
            go.Scatter(
                x=self.obs_X[dim_name].values,
                y=self.obs_y.values,
                mode='markers',
                marker=dict(
                    size=[float(s * 20 + 5) for s in point_size],
                    color=colors,
                    line=dict(width=1, color='black'),
                    opacity=0.7
                ),
                hovertext=hover_text,
                hoverinfo='text',
                name='Observations',
                showlegend=False
            ),
            row=ax.row, col=ax.col
        )

    def plot_all(self, coordinates: list[float] | Tensor | pd.Series, linspace=None, figsize=(24, 12), render = False):
        """Handle plotting all dimensions using Plotly subplots."""
        
        if coordinates is None:
            coordinates = self.get_best_observed_coord()

        # Normalize coordinates into a torch.Tensor
        if isinstance(coordinates, pd.Series):
            coordinates = torch.tensor(coordinates.values, dtype=dtype)
        elif isinstance(coordinates, np.ndarray):
            coordinates = torch.tensor(coordinates, dtype=dtype)

        self.fig, axs = self._create_subplots(figsize=figsize)

        linspace = self._create_linspace() if linspace is None else linspace
        
        for i in range(self.obs_X.shape[1]):
            ax = axs[i]

            if not isinstance(linspace[i], Tensor):
                grid = torch.tensor(linspace[i], dtype=dtype, device=device)
            else:
                grid = linspace[i].detach().clone()

            mean, std = self._get_plane_gaussian_for_point(
                coordinates, fixed_dim=i, grid=grid
            )

            self._plot_gp(grid, mean, std, ax, coordinates)
            dim = self.obs_X.columns[i]

            self._vlines(ax, coordinates[i])
            self._plot_observations(ax, dim, coordinates)
            self._add_expected_improvement(ax, dim, coordinates)

        rounded_coords = [f"{x:.3g}" for x in coordinates]
        self._add_subplot_elements(rounded_coords)
        self.fig.show() if render else None

        return self.fig, axs

    def _add_expected_improvement_plotly(self, fig, dim: str, coordinates: Tensor, row, col):
        mean, std = self._eval_gp(self.predict_X)
        dim_x = self.predict_X.loc[:, dim]
        sizes = self._get_size(self.predict_X, coordinates, dim)
        self._plot_expected_improvement(fig, dim_x, mean, std, sizes, row, col)

    def return_fig(self):
        return self.fig

    def show(self):
        """Display the Plotly figure."""
        self.fig.show()

if __name__ == "__main__":

    with open("data/example_manager.pkl", "rb") as f:
        manager = BayesClientManager.init_self_from_pickle(f)

    visualiser = GPVisualiserPlotly(
        bayes_manager=manager,
    )
    fig, axs = visualiser.plot_all(coordinates=None)

    fig.show(render="browser")


    # Explicitly write figure to an ephemeral HTML file and open in default browser
    _html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as _f:
        _f.write(_html)
    webbrowser.open("file://" + _f.name)
    print("done")
