"""
Plotly visualization components for ECG signals, scalograms, 3D tensors, and Tucker cores.
"""

from typing import List, Optional

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None

from core.config import FS, LEAD_NAMES


def _check_plotly():
    if go is None or make_subplots is None:
        raise ImportError(
            "plotly is required for visualization functions. "
            "Install it using: pip install plotly"
        )


def plot_ecg(
    signal: np.ndarray,
    title: str = "12-Lead ECG",
    lead_names: Optional[List[str]] = None,
    fs: int = FS,
):
    """
    Generate an interactive 6×2 multi-panel Plotly figure displaying all 12 ECG leads.
    """
    _check_plotly()
    names = lead_names or LEAD_NAMES
    times = np.arange(signal.shape[0]) / fs

    fig = make_subplots(
        rows=6,
        cols=2,
        subplot_titles=names[: signal.shape[1]],
        vertical_spacing=0.04,
        horizontal_spacing=0.07,
    )

    for index in range(min(12, signal.shape[1])):
        row = index // 2 + 1
        col = index % 2 + 1

        fig.add_trace(
            go.Scattergl(
                x=times,
                y=signal[:, index],
                mode="lines",
                name=names[index],
                showlegend=False,
                line=dict(width=1),
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(title_text="Time (s)", row=row, col=col)
        fig.update_yaxes(title_text="mV", row=row, col=col)

    fig.update_layout(
        title=title,
        height=950,
        margin=dict(l=25, r=25, t=60, b=25),
    )
    return fig


def plot_scalogram(
    tensor: np.ndarray,
    frequencies: np.ndarray,
    lead_index: int = 0,
    lead_name: Optional[str] = None,
    fs: int = FS,
):
    """
    Generate a 2D Time-Frequency scalogram heatmap for a specified ECG lead.
    """
    _check_plotly()
    name = lead_name or (LEAD_NAMES[lead_index] if lead_index < len(LEAD_NAMES) else f"Lead {lead_index+1}")
    times = np.arange(tensor.shape[0]) / fs

    fig = go.Figure(
        data=go.Heatmap(
            z=tensor[:, :, lead_index].T,
            x=times,
            y=frequencies,
            colorscale="Viridis",
            colorbar=dict(title="Magnitude"),
        )
    )

    fig.update_layout(
        title=f"CWT Scalogram — Lead {name}",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        height=480,
        margin=dict(l=25, r=25, t=50, b=25),
    )
    return fig


def plot_3d_tensor(
    tensor: np.ndarray,
    frequencies: np.ndarray,
    time_step: int = 25,
    lead_names: Optional[List[str]] = None,
    fs: int = FS,
    max_points: int = 18000,
):
    """
    Generate an interactive 3D point cloud of the (Time × Frequency × Lead) tensor.
    Downsamples along time for fluid interactive rendering.
    """
    _check_plotly()
    names = lead_names or LEAD_NAMES
    sampled = tensor[::time_step, :, :]
    time_values = np.arange(0, tensor.shape[0], time_step) / fs

    time_grid, freq_grid, lead_grid = np.meshgrid(
        time_values,
        frequencies,
        np.arange(tensor.shape[2]),
        indexing="ij",
    )

    x = time_grid.ravel()
    y = freq_grid.ravel()
    z = lead_grid.ravel()
    magnitude = sampled.ravel()

    if magnitude.size > max_points:
        indices = np.linspace(0, magnitude.size - 1, max_points, dtype=int)
        x = x[indices]
        y = y[indices]
        z = z[indices]
        magnitude = magnitude[indices]

    fig = go.Figure(
        data=go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=2,
                color=magnitude,
                colorscale="Viridis",
                opacity=0.75,
                colorbar=dict(title="Magnitude"),
            ),
            hovertemplate=(
                "Time: %{x:.3f} s<br>"
                "Frequency: %{y:.2f} Hz<br>"
                "Lead index: %{z}<br>"
                "Magnitude: %{marker.color:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Interactive 3D CWT Tensor Volume",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            zaxis=dict(
                title="Lead",
                tickmode="array",
                tickvals=list(range(min(12, tensor.shape[2]))),
                ticktext=names[: tensor.shape[2]],
            ),
            aspectmode="manual",
            aspectratio=dict(x=2, y=1.2, z=1),
        ),
        height=680,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_core_slice(
    core: np.ndarray,
    axis: str = "Time rank × Frequency rank",
):
    """
    Generate a 2D heatmap showing an orthogonal 2D slice through the Tucker core tensor.
    """
    _check_plotly()
    if axis == "Time rank × Frequency rank":
        matrix = core[:, :, 0]
        x_title = "Frequency rank"
        y_title = "Time rank"
    elif axis == "Time rank × Lead rank":
        matrix = core[:, 0, :]
        x_title = "Lead rank"
        y_title = "Time rank"
    else:
        matrix = core[0, :, :]
        x_title = "Lead rank"
        y_title = "Frequency rank"

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            colorscale="Viridis",
            colorbar=dict(title="Core value"),
        )
    )

    fig.update_layout(
        title=f"Tucker Core Slice — {axis}",
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=450,
        margin=dict(l=25, r=25, t=50, b=25),
    )
    return fig

