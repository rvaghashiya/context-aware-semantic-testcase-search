"""
Interactive visualization using Plotly.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def create_cluster_plot(coordinates: np.ndarray, labels: np.ndarray, 
                       hover_text: List[str], title: str = "Semantic Clusters",
                       show_plot: bool = True) -> Optional[Any]:
    """Create interactive cluster visualization."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    if coordinates.shape[1] == 2:
        return create_2d_cluster_plot(coordinates, labels, hover_text, title, show_plot)
    elif coordinates.shape[1] == 3:
        return create_3d_cluster_plot(coordinates, labels, hover_text, title, show_plot)
    else:
        raise ValueError("Coordinates must be 2D or 3D")

def create_2d_cluster_plot(coordinates: np.ndarray, labels: np.ndarray,
                          hover_text: List[str], title: str = "2D Semantic Clusters",
                          show_plot: bool = True) -> Optional[Any]:
    """Create 2D interactive cluster plot."""
    if not PLOTLY_AVAILABLE:
        return None

    df = pd.DataFrame({
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'cluster': labels,
        'text': hover_text
    })

    fig = px.scatter(df, x='x', y='y', color='cluster', hover_name='text',
                    title=title, color_continuous_scale='viridis')

    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        width=800, height=600,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2"
    )

    if show_plot:
        fig.show()

    return fig

def create_3d_cluster_plot(coordinates: np.ndarray, labels: np.ndarray,
                          hover_text: List[str], title: str = "3D Semantic Clusters",
                          show_plot: bool = True) -> Optional[Any]:
    """Create 3D interactive cluster plot."""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    # Get unique clusters and colors
    unique_labels = np.unique(labels)
    colors = px.colors.qualitative.Set1[:len(unique_labels)]

    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        cluster_coords = coordinates[mask]
        cluster_texts = [hover_text[j] for j in np.where(mask)[0]]

        fig.add_trace(go.Scatter3d(
            x=cluster_coords[:, 0],
            y=cluster_coords[:, 1], 
            z=cluster_coords[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=colors[i % len(colors)],
                opacity=0.7
            ),
            text=cluster_texts,
            hovertemplate='<b>%{text}</b><br>Cluster: ' + str(cluster_id),
            name=f'Cluster {cluster_id}'
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2", 
            zaxis_title="Dimension 3"
        ),
        width=900,
        height=700
    )

    if show_plot:
        fig.show()

    return fig

def create_similarity_heatmap(similarity_matrix: np.ndarray, 
                             text_ids: List[str],
                             title: str = "Similarity Heatmap") -> Optional[Any]:
    """Create similarity heatmap."""
    if not PLOTLY_AVAILABLE:
        return None

    # Sample for performance if too large
    if len(text_ids) > 50:
        indices = np.random.choice(len(text_ids), 50, replace=False)
        similarity_matrix = similarity_matrix[indices][:, indices] 
        text_ids = [text_ids[i] for i in indices]

    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=text_ids,
        y=text_ids,
        colorscale='viridis'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Test Cases",
        yaxis_title="Test Cases",
        width=800,
        height=800
    )

    return fig

def save_plot_as_html(fig, filename: str):
    """Save plot as HTML file."""
    if fig is not None and PLOTLY_AVAILABLE:
        plot(fig, filename=filename, auto_open=False)
        print(f"Plot saved as {filename}")
