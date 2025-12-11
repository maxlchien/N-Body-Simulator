
import os
import numpy as np
from CSV_processor import csv_processor
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

def plotter(csv_file, color=None, save_as="nbody_animation.mp4"):
    """
    Plot an N-body simulation from a CSV file.

    Parameters
    ----------
    csv_file : str
        The CSV results file (filename).
    colors : list, optional
        List of colors corresponding to each body.
    save_as : str, optional
        Output animation filename.
    """

    G, dt, num_bodies, df = csv_processor(csv_file)

    # Reset index so animation frames match rows
    df = df.reset_index(drop=True)

    # Normalize values for visualization
    scale_list = []
    for n in range(num_bodies):
        scale_list.append(df[f"x{n+1}"].abs().max())
        scale_list.append(df[f"y{n+1}"].abs().max())
    scale = max(scale_list)

    for n in range(num_bodies):
        df[f"x{n+1}n"] = df[f"x{n+1}"] / scale
        df[f"y{n+1}n"] = df[f"y{n+1}"] / scale

    # Setting up plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_axis_off()

    title = ax.text(
        0.70,
        1.05,
        "",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        fontfamily="DejaVu Serif",
    )
    
    const_text = ax.text(
        0,
        1.05,
        f"G = {G}",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        fontfamily="DejaVu Serif",
    )

    # Generate N random stars for background
    N_stars = 500
    star_x = np.random.uniform(-1.1, 1.1, N_stars)
    star_y = np.random.uniform(-1.1, 1.1, N_stars)
    ax.scatter(star_x, star_y, color="white", s=1, alpha=0.7, zorder=0)

    bodies = []
    trail = []
    for n in range(num_bodies):
        if color is None:
            bodies.append(ax.scatter([], [], s=150))
        else:
            bodies.append(ax.scatter([], [], color=color[n], s=150))

        line, = ax.plot([], [], color="white", linewidth=1)
        trail.append(line)

    # Updating frames
    def update(frame):
        for n in range(num_bodies):
            bodies[n].set_offsets(
                [
                    [
                        df.iloc[frame][f"x{n+1}n"],
                        df.iloc[frame][f"y{n+1}n"],
                    ]
                ]
            )
            trail[n].set_data(
                df[f"x{n+1}n"][:frame],
                df[f"y{n+1}n"][:frame],
            )
            current_time = frame * dt
            title.set_text(f"Time = {current_time:.2f}")

        return bodies + trail

    # Animation
    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=10,
        blit=False,
    )
    plt.close(fig)


    anim.save(save_as, writer="ffmpeg", fps=50)

    
    return f"Saved animation to {save_as}"
