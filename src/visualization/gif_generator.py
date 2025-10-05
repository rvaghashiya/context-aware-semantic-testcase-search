"""
GIF generation for animated visualizations.
"""
import numpy as np
from typing import List, Optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from PIL import Image
    import io
    ANIMATION_AVAILABLE = True
except ImportError:
    ANIMATION_AVAILABLE = False

def create_clustering_evolution_gif(coordinates_list: List[np.ndarray],
                                   labels_list: List[np.ndarray],
                                   titles: List[str],
                                   output_path: str = "clustering_evolution.gif",
                                   duration: int = 1000):
    """Create GIF showing clustering evolution."""
    if not ANIMATION_AVAILABLE:
        print("Animation libraries not available")
        return

    frames = []

    for coords, labels, title in zip(coordinates_list, labels_list, titles):
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, 
                           cmap='tab10', alpha=0.7, s=50)
        ax.set_title(title)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)

        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()

    # Save as GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    print(f"GIF saved to {output_path}")

def create_parameter_sweep_gif(coordinates: np.ndarray,
                              parameter_values: List[float],
                              parameter_name: str,
                              clustering_func,
                              output_path: str = "parameter_sweep.gif"):
    """Create GIF showing effect of parameter changes."""
    if not ANIMATION_AVAILABLE:
        return

    frames = []

    for param_val in parameter_values:
        # Apply clustering with current parameter
        if parameter_name == 'n_clusters':
            labels = clustering_func(coordinates, n_clusters=int(param_val))
        elif parameter_name == 'eps':
            labels = clustering_func(coordinates, eps=param_val)
        else:
            labels = clustering_func(coordinates)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                           c=labels, cmap='tab10', alpha=0.7, s=50)

        ax.set_title(f'{parameter_name} = {param_val}')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()

    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=800, loop=0)
    print(f"Parameter sweep GIF saved to {output_path}")
