"""
ui_components/dual_visualizer.py
===================================
Wrappers that render, side by side in the left "Live Matrix" panel, the
mandated triple output for every module:
    1. Matplotlib line chart (exact gradients / drop-off points)
    2. Seaborn heatmap (2D spatial contours / hotspots)
    3. Pandas DataFrame (raw extractable numerical matrix)

Pure presentation. Physics arrays are passed in already computed.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def render_line_chart(x_vals: np.ndarray, y_vals: np.ndarray, *, title: str,
                       xlabel: str, ylabel: str, marker: str = "o") -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(x_vals, y_vals, marker=marker, markersize=3, linewidth=1.6, color="#1c4f7c")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_heatmap(field_2d: np.ndarray, *, title: str, cbar_label: str,
                    cmap: str = "coolwarm", center: float | None = 0.0) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        field_2d,
        cmap=cmap,
        center=center,
        cbar_kws={"label": cbar_label},
        ax=ax,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_dataframe(field_2d: np.ndarray, x: np.ndarray, y: np.ndarray, *,
                      value_name: str = "T (°C)", decimals: int = 2,
                      max_rows_display: int = 200) -> pd.DataFrame:
    """
    Build and render the raw numerical matrix as a Pandas DataFrame
    (rows = y-coordinates, columns = x-coordinates), with a download button
    for the full-resolution CSV.
    """
    df = pd.DataFrame(
        np.round(field_2d, decimals),
        index=np.round(y, 3),
        columns=np.round(x, 3),
    )
    df.index.name = "y [m]"
    df.columns.name = "x [m]"

    st.caption(f"Raw matrix — {value_name} (rows: y, columns: x)")
    st.dataframe(df, width="stretch", height=280)

    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label=f"Download full-resolution matrix (CSV) — {value_name}",
        data=csv,
        file_name=f"{value_name.replace(' ', '_').replace('(', '').replace(')', '')}_matrix.csv",
        mime="text/csv",
    )
    return df


def render_dual_visualization(
    *,
    line_x: np.ndarray,
    line_y: np.ndarray,
    line_title: str,
    line_xlabel: str,
    line_ylabel: str,
    field_2d: np.ndarray,
    heatmap_title: str,
    heatmap_cbar_label: str,
    x: np.ndarray,
    y: np.ndarray,
    df_value_name: str,
) -> None:
    """Convenience wrapper: renders line chart + heatmap side by side, then the DataFrame."""
    col_a, col_b = st.columns(2)
    with col_a:
        render_line_chart(line_x, line_y, title=line_title, xlabel=line_xlabel, ylabel=line_ylabel)
    with col_b:
        render_heatmap(field_2d, title=heatmap_title, cbar_label=heatmap_cbar_label)
    render_dataframe(field_2d, x, y, value_name=df_value_name)
