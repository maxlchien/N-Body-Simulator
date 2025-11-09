from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation


def plotter(filename):
    # Extracting the positions of each body at each time step
    df = pd.read_csv(filename)
    bodies = df["bodies"].unique()
    time = df["time"].unique()
    data = {}
    for body in bodies:
        new_df = df[df["bodies"] == body][["px", "py"]]
        data[body] = new_df.to_numpy()

    # Plotting the body in initial state
    fig, ax = plt.subplots()
    plots = {}
    for body in bodies:
        px_0, py_0 = data[body][0]
        plots[body] = ax.plot([px_0], [py_0], marker="o")[0]

    ax.set_xlim(df["px"].min() - 0.5, df["px"].max() + 0.5)
    ax.set_ylim(df["py"].min() - 0.5, df["py"].max() + 0.5)
    ax.set_aspect("equal")

    # Update for animation
    def update(frame):
        for body in bodies:
            px, py = data[body][frame]
            plots[body].set_data([px], [py])
        return list(plots.values())

    ani = FuncAnimation(fig=fig, func=update, frames=len(time), interval=50, blit=False)
    ani.save("nbody_animation.mp4", writer="ffmpeg", fps=10)
