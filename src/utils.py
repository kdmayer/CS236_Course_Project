import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np
import plotly as plt


def plot_samples(samples, num=8, rows=2, cols=4):
    fig = plt.subplots.make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "Scatter3d"} for _ in range(cols)] for _ in range(rows)],
    )
    indices = torch.randperm(samples.size(0))[:num]
    for i, sample in enumerate(samples[indices].cpu()):
        fig.add_trace(
            plt.graph_objects.Scatter3d(
                x=sample[:, 0],
                y=sample[:, 2],
                z=sample[:, 1],
                mode="markers",
                marker=dict(size=3, opacity=0.8),
            ),
            row=i // cols + 1,
            col=i % cols + 1,
        )
    fig.update_layout(showlegend=False)
    return fig

def normalize_pc(point_cloud):

    # Find the range for each axis and then the max range
    mins = np.min(point_cloud, axis=0)
    maxs = np.max(point_cloud, axis=0)
    ranges = maxs - mins
    max_range = np.max(ranges)

    # Handle the case where max range is 0 (to prevent division by zero)
    if max_range == 0:
        raise ValueError("Point cloud has zero range.")

    # Normalize the point cloud using the same scale for all axes and then
    # shift to [-1, 1]
    pc_normalized = 2 * (point_cloud - mins) / max_range - 1

    # Now we want everything to fit in -1,1, but without touching these bounds.
    pc_normalized = pc_normalized * 0.999
    pc_normalized = pc_normalized.astype(np.float32)

    return pc_normalized