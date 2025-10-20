import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd


class MeshGridSlicePlotter:
    """Compact slice-contour plotter with a single shared colorbar.

    Usage: MeshGridSlicePlotter(df, parameters, response_label, center_coords, empirical_grid_df=...).plot_all_slices()
    """

    def __init__(
        self,
        meshgrid_df,
        parameters,
        response_label,
        center_coords,
        cutoff=50,
        title="Parameter Slice Contours",
        empirical_grid_df=None,
    ):
        self.df = meshgrid_df
        self.parameters = list(parameters)
        self.response = response_label
        self.center = center_coords
        self.cutoff = cutoff
        self.empirical = empirical_grid_df
        self.title = title
        # pairs and fixed grid layout (2x3 for 4 params)
        self.pairs = [
            (p1, p2)
            for i, p1 in enumerate(self.parameters)
            for p2 in self.parameters[i + 1 :]
        ]
        self.nrows, self.ncols = 2, 3
        # reduce horizontal/vertical spacing between subplots to waste less space
        self.fig = sp.make_subplots(
            rows=self.nrows,
            cols=self.ncols,
            subplot_titles=[f"{a} vs {b}" for a, b in self.pairs],
            horizontal_spacing=0.04,
            vertical_spacing=0.08,
        )
        # global color scale
        if self.response in getattr(self.df, "columns", []):
            self.zmin, self.zmax = float(self.df[self.response].min()), float(
                self.df[self.response].max()
            )
        else:
            self.zmin = self.zmax = None
        # legend control: ensure we only add each legend item once
        self._legend_empirical_points_added = False
        self._legend_empirical_max_added = False

    def _mask(self, fixed):
        return self.df[
            np.logical_and.reduce(
                [np.isclose(self.df[p], v, atol=self.cutoff) for p, v in fixed.items()]
            )
        ].copy()

    def _add_marker(self, subset, x, y, value_col, **kw):
        """Add a labeled marker at the max value of `value_col` (if present) or first row.
        `row` and `col` must be provided as keyword args for subplot placement.
        """
        if subset.empty:
            return
        if value_col in subset.columns:
            try:
                sel = subset.loc[subset[value_col].idxmax()]
            except Exception:
                sel = subset.iloc[0]
        else:
            sel = subset.iloc[0]

        r = kw.get("row")
        c = kw.get("col")
        marker = dict(
            color=kw.get("color", "red"),
            size=kw.get("size", 10),
            symbol=kw.get("symbol", "circle"),
        )
        text = kw.get("text", "")
        textpos = kw.get("textpos", "top center")

        # add trace to specified subplot
        # add trace to specified subplot; show legend only for the first such trace
        showleg = False
        if not self._legend_empirical_max_added:
            showleg = True
            self._legend_empirical_max_added = True
        self.fig.add_trace(
            go.Scatter(
                x=[sel[x]],
                y=[sel[y]],
                mode="markers+text",
                name="Empirical max",
                text=[text],
                textposition=textpos,
                marker=marker,
                showlegend=showleg,
            ),
            row=r,
            col=c,
        )

    def _add_empirical(self, subset, x, y, row, col):
        if subset is None or subset.empty:
            return
        showleg = False
        if not self._legend_empirical_points_added:
            showleg = True
            self._legend_empirical_points_added = True
        self.fig.add_trace(
            go.Scatter(
                x=subset[x],
                y=subset[y],
                mode="markers",
                name="Empirical observations",
                marker=dict(color="black", size=6),
                opacity=0.7,
                showlegend=showleg,
            ),
            row=row,
            col=col,
        )

    def mark_bayes_investigated_boinths(self, df):
        """Accept the dataframe with investigated points (may have other columns; only param columns are kept)."""

        if df is None:
            self.bayes_investigated = pd.DataFrame(columns=self.parameters)
            return self.bayes_investigated

        # keep only the columns that match our parameter names; ignore the rest
        filtered = (
            df.reindex(columns=self.parameters).dropna(how="all").reset_index(drop=True)
        )
        self.bayes_investigated = filtered
        return filtered

    def plot_parameter_contour(self, x, y, fixed, row, col):
        subset = self._mask(fixed)
        if subset.empty:
            return
        subset = subset.drop_duplicates(subset=[x, y])
        X = np.sort(subset[x].unique())
        Y = np.sort(subset[y].unique())
        Z = subset.pivot(index=y, columns=x, values=self.response).values
        if Z.size == 0 or Z.shape[0] < 2 or Z.shape[1] < 2:
            return

        # keep the colorbar compact and place it just outside the subplots to avoid large right margin
        cb = dict(
            title={"text": self.response},
            len=0.9,
            y=0.5,
            yanchor="middle",
            thickness=15,
            x=1.02,
        )
        show_cb = row == self.nrows and col == self.ncols
        self.fig.add_trace(
            go.Contour(
                z=Z,
                x=X,
                y=Y,
                colorscale="Viridis",
                zmin=self.zmin,
                zmax=self.zmax,
                colorbar=cb if show_cb else None,
                contours_coloring="heatmap",
                showscale=show_cb,
                name="Limonene (mg/L)" if self.response == "pred_response" else "Noise level (mg/L)",
                legendgroup="Limonene (mg/L)" if self.response == "pred_response" else "Noise level (mg/L)"
            ),
            row=row,
            col=col,
        )
        # markers
        self._add_marker(
            subset,
            x,
            y,
            self.response,
            row=row,
            col=col,
            color="red",
            text="Empirically observed maximum",
            textpos="bottom center",
            size=12,
        )
        self._add_empirical(self.empirical, x, y, row, col)


        x_min, x_max = float(np.min(X)), float(np.max(X))
        y_min, y_max = float(np.min(Y)), float(np.max(Y))
        pad_x = 0.01 * (x_max - x_min) if x_max > x_min else 0
        pad_y = 0.01 * (y_max - y_min) if y_max > y_min else 0
        self.fig.update_xaxes(range=[x_min - pad_x, x_max + pad_x], autorange=False, row=row, col=col)
        self.fig.update_yaxes(range=[y_min - pad_y, y_max + pad_y], autorange=False, row=row, col=col)

    def plot_all_slices(self):
        for idx, (x, y) in enumerate(self.pairs):
            fixed = {p: self.center[p] for p in self.parameters if p not in (x, y)}
            r, c = (idx // self.ncols) + 1, (idx % self.ncols) + 1
            self.plot_parameter_contour(x, y, fixed, r, c)
            # reduce distance between axis titles and the axis/plot
            self.fig.update_xaxes(
                title_text=x, title_standoff=6, ticks="inside", ticklen=4, row=r, col=c
            )
            self.fig.update_yaxes(
                title_text=y, title_standoff=6, ticks="inside", ticklen=4, row=r, col=c
            )

        # tighter overall margins so the subplots and axis titles sit closer to the figure edges
        self.fig.update_layout(
            height=900,
            width=1400,
            title={"text": self.title, "x": 0.5},
            showlegend=True,
            legend=dict(title="Legend", orientation="v", yanchor="bottom", y=0.95),
            margin=dict(l=40, r=120, t=100, b=40),
        )
        return self.fig

