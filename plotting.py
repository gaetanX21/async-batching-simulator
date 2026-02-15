from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from simulation import SimulationConfig

OutputMode = Literal["save", "show", "return"]

SAVE_DIR = Path(__file__).parent / "plots"
SAVE_DIR.mkdir(exist_ok=True)


def plot_batcher_history(
    batcher_history: Sequence[tuple[float, Sequence[int]]],
    config: "SimulationConfig",
    output: OutputMode = "save",
) -> Figure | None:
    """Plot the composition of SHORT vs LONG sequences in each batch.

    Args:
        batcher_history: List of (timestamp, sequences) tuples where sequences is a list of sequence lengths
        config: Simulation configuration containing distribution parameters
        output: Output mode - "save" to save to file, "show" to display, "return" to return figure
    """
    if not batcher_history:
        print("No batcher history to plot")
        return None

    short = config.dist.short
    long = config.dist.long

    num_batches = len(batcher_history)
    short_counts = []
    long_counts = []

    # Count SHORT and LONG sequences in each batch
    for _timestamp, batch in batcher_history:
        counts = Counter(batch)
        short_counts.append(counts.get(short, 0))
        long_counts.append(counts.get(long, 0))

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Stacked bar chart
    x = np.arange(num_batches)
    width = 0.8

    ax.bar(
        x,
        short_counts,
        width,
        label=f"SHORT ({short}s)",
        color="#3498db",
        alpha=0.8,
    )
    ax.bar(
        x,
        long_counts,
        width,
        bottom=short_counts,
        label=f"LONG ({long}s)",
        color="#e74c3c",
        alpha=0.8,
    )

    # Add expected number of short sequences per batch as horizontal line
    expected_short = config.num_short_per_batch
    ax.axhline(
        y=expected_short,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Expected SHORT ({expected_short:.1f})",
        alpha=0.7,
    )

    ax.set_xlabel("Batch Index", fontsize=12)
    ax.set_ylabel("Num Seq", fontsize=12)
    ax.set_title(
        "Batch Composition: SHORT vs LONG Sequences", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks([])  # avoid cluster

    plt.tight_layout()

    if output == "save":
        output_path = SAVE_DIR / "batcher_history.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)
        return None
    elif output == "show":
        plt.show()
        return None
    else:  # output == "return"
        return fig


def plot_inference_engine_history(
    inference_engine_history: Sequence[tuple[float, Sequence]],
    config: "SimulationConfig",
    output: OutputMode = "save",
) -> Figure | None:
    """Plot inference engine content and lane timeline in a single figure with shared x-axis.

    Args:
        inference_engine_history: List of (timestamp, lanes) tuples where lanes is a list of items (or None for empty lanes)
        config: Simulation configuration containing distribution parameters
        output: Output mode - "save" to save to file, "show" to display, "return" to return figure
    """
    if not inference_engine_history:
        print("No inference engine history to plot")
        return None

    short = config.dist.short
    long = config.dist.long

    # Extract timestamps and counts for SHORT and LONG at each time step
    timestamps = []
    short_counts = []
    long_counts = []
    lane_states = []

    for timestamp, lanes in inference_engine_history:
        timestamps.append(timestamp)

        # Count SHORT and LONG sequences in the lanes
        short_count = 0
        long_count = 0
        state = []

        for item in lanes:
            if item is None:
                state.append(0)  # Empty
            else:
                seq_len = item.seq if hasattr(item, "seq") else item
                if seq_len == short:
                    short_count += 1
                    state.append(1)  # Short
                elif seq_len == long:
                    long_count += 1
                    state.append(2)  # Long
                else:
                    state.append(0)  # Unknown, treat as empty

        short_counts.append(short_count)
        long_counts.append(long_count)
        lane_states.append(state)

    # Convert timestamps to relative time
    start_time = timestamps[0]
    x = np.array([t - start_time for t in timestamps])

    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), dpi=150, sharex=True, height_ratios=[1, 2]
    )

    # ===== TOP PLOT: Inference Engine Content Over Time =====
    # Stacked area plot
    ax1.fill_between(
        x, 0, short_counts, label=f"SHORT ({short}s)", color="#3498db", alpha=0.6
    )
    ax1.fill_between(
        x,
        short_counts,
        np.array(short_counts) + np.array(long_counts),
        label=f"LONG ({long}s)",
        color="#e74c3c",
        alpha=0.6,
    )

    # Add total line
    total_counts = np.array(short_counts) + np.array(long_counts)
    ax1.plot(
        x,
        total_counts,
        color="black",
        linewidth=2,
        label="Total",
        linestyle="-",
        alpha=0.7,
    )

    # Add Little's law prediction for SHORT sequences
    expected_short_load = config.short_load
    ax1.axhline(
        y=expected_short_load,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Expected SHORT ({expected_short_load:.1f})",
        alpha=0.8,
    )

    ax1.set_ylabel("Num Seq inside Engine", fontsize=12)
    ax1.set_title(
        "Inference Engine: Content Over Time and Lane Occupancy Timeline",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(axis="both", alpha=0.3)

    # Add statistics as text
    max_total = max(total_counts)
    avg_total = np.mean(total_counts)
    duration = x[-1] if len(x) > 0 else 0
    ax1.text(
        0.02,
        0.98,
        f"Max concurrent: {max_total} req\nAvg concurrent: {avg_total:.1f} req\nDuration: {duration:.1f}s",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # ===== BOTTOM PLOT: Lane Occupancy Timeline =====
    num_lanes = len(lane_states[0]) if lane_states else 0
    if num_lanes == 0:
        print("No lanes to plot")
        return None

    # Create 2D grid for visualization
    grid = np.array(lane_states).T  # Transpose to get [lane_idx][time_idx]

    # Create custom colormap: white (empty), blue (short), red (long)
    from matplotlib.colors import ListedColormap

    colors = ["white", "#3498db", "#e74c3c"]  # empty, short, long
    cmap = ListedColormap(colors)

    # Plot as heatmap
    ax2.imshow(
        grid,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        extent=(x[0], x[-1], num_lanes - 0.5, -0.5),
        vmin=0,
        vmax=2,
        alpha=0.6,
    )

    # Set labels
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Lane Index", fontsize=12)

    # Set y-axis to show lane indices
    ax2.set_yticks(np.arange(0, num_lanes, max(1, num_lanes // 20)))  # Show ~20 ticks
    ax2.grid(False)

    # Add statistics with color legend
    total_snapshots = len(lane_states)
    occupancy_rate = (grid > 0).sum() / (num_lanes * total_snapshots) * 100
    ax2.text(
        0.02,
        0.98,
        f"Lanes: {num_lanes}\nAvg occupancy: {occupancy_rate:.1f}%",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if output == "save":
        output_path = SAVE_DIR / "inference_engine_combined.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)
        return None
    elif output == "show":
        plt.show()
        return None
    else:  # output == "return"
        return fig
