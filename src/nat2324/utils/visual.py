import os
from typing import Any, Callable, Collection

import matplotlib.pyplot as plt
import numpy as np


@plt.style.context("bmh")
def visualize_objectives_3d(
    objectives: dict[str, tuple[Callable[[np.ndarray], float], tuple[float, float]]],
    num_points: int = 100,
    max_col: int = 5,
    filepath: str | None = None,
    plot_surface_kwargs: dict[str, Any] = {},
):
    # Get the function names and calculate the number of rows and columns
    function_names = list(objectives.keys())
    n_cols = min(len(function_names), max_col)
    n_rows = len(function_names) // n_cols + (len(function_names) % n_cols > 0)

    # Create the figure and set the layout to be tight
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4), tight_layout=True)

    # Set the default keyword arguments for the plot_surface function
    plot_surface_kwargs.setdefault("rstride", 1)
    plot_surface_kwargs.setdefault("cstride", 1)
    plot_surface_kwargs.setdefault("cmap", "viridis")
    plot_surface_kwargs.setdefault("edgecolor", "none")

    for i, function_name in enumerate(function_names, 1):
        # Unpack the function callback and bounds
        function, bounds = objectives[function_name]

        # Create the meshgrid and calculate the function values
        x = y = np.linspace(*bounds, num_points)
        X, Y = np.meshgrid(x, y)
        Z = np.apply_along_axis(function, -1, np.dstack([X, Y]))

        # Create the subplot and plot the surface; set the title
        ax = fig.add_subplot(n_rows, n_cols, i, projection="3d")
        ax.plot_surface(X, Y, Z, **plot_surface_kwargs)
        ax.set_title(function_name, fontsize=20)

    if filepath is not None:
        # Save the figure if a filepath is provided
        plt.savefig(filepath, dpi=300)

    # Show plot
    plt.show()


@plt.style.context("bmh")
def visualize_optimization_experiments(
    xs: dict[str, np.ndarray],
    ys: dict[str, np.ndarray],
    zs: dict[str, np.ndarray] | None = None,
    curve_labels: str | list[str | list[str]] | None = None,
    labels: str | list[str | list[str]] | None = None,
    titles: str | list[str] | None = None,
    max_cols: int = 3,
    is_maximization: bool | None = None,
    filepath: str | None = None,
    scale: tuple[int, int] = (4, 4),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ignore_suffix_x: bool = False,
    ignore_suffix_y: bool = False,
    ignore_suffix_z: bool = False,
    elev: float = 30.0,
    azim: float = -150.0,
):
    # Convert the dictionary values to numpy arrays
    xs = {k: np.array(v) for k, v in xs.items()}
    ys = {k: np.array(v) for k, v in ys.items()}
    zs = {k: np.array(v) for k, v in zs.items()} if zs is not None else None

    # if is_reverse:
    #     # Reverse the order of the arrays
    #     xs = {k: v[..., ::-1] for k, v in xs.items()}

    #     if zs is None:
    #         ys = {k: v[..., ::-1, :] for k, v in ys.items()}
    #     else:
    #         ys = {k: v[..., ::-1] for k, v in ys.items()}
    #         zs = {k: v[..., ::-1, :] for k, v in zs.items()}

    if curve_labels is not None and (
        isinstance(curve_labels, str) or not isinstance(curve_labels[0], list)
    ):
        # Init new labels
        new_labels = []

        for key in ys.keys():
            if isinstance(curve_labels, list):
                # Assume dims of same size
                new_labels.append(curve_labels)
            elif isinstance(np.array(ys[key])[0].tolist(), (int, float)):
                new_labels.append([curve_labels])
            else:
                new_labels.append([curve_labels] * len(ys[key][0]))

        # Convert to a list of lists
        curve_labels = new_labels

    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust subplot parameters
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2, left=-0.2, right=0.8, top=0.8)
    x_keys = list(xs.keys())
    y_keys = list(ys.keys())

    if zs is not None:
        x_keys = x_keys * len(zs) if len(x_keys) == 1 else x_keys
        y_keys = y_keys * len(zs) if len(y_keys) == 1 else y_keys

        if len(zs) > 1 and titles is None:
            titles = list(zs.keys())
        elif labels is None:
            labels = list(
                z_key.split("_")[0] if ignore_suffix_z else z_key for z_key in zs.keys()
            )

    else:
        x_keys = x_keys * len(ys) if len(x_keys) == 1 else x_keys

        if len(ys) > 1 and titles is None:
            titles = list(ys.keys())
        elif labels is None:
            labels = list(
                y_key.split("_")[0] if ignore_suffix_y else y_key for y_key in ys.keys()
            )

    labels = (
        [labels] * len(x_keys)
        if labels is not None and isinstance(labels, str)
        else labels
    )

    titles = (
        [titles] * len(x_keys)
        if titles is not None and isinstance(titles, str)
        else titles
    )

    n_cols = min(len(x_keys), max_cols)
    n_rows = len(x_keys) // n_cols + (len(x_keys) % n_cols > 0)
    fig = plt.figure(
        figsize=(n_cols * scale[0], n_rows * scale[1]), layout="constrained"
    )

    for i, (x_key, y_key) in enumerate(zip(x_keys, y_keys)):
        if zs is not None:
            # plt.cla()  # clear current axes
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect(aspect=None, zoom=0.85)

            X, Y = np.meshgrid(xs[x_key], ys[y_key])
            z_key = list(zs.keys())[i]

            for j in range(zs[z_key].shape[-1]):
                Z = np.moveaxis(zs[z_key][:, :, j], 0, 1)
                ax.plot_surface(
                    X,
                    Y,
                    Z,
                    label=curve_labels[i][j] if curve_labels else None,
                    alpha=0.7,
                    cmap="viridis",
                    cstride=1,
                    rstride=1,
                )

                if is_maximization is not None:
                    max_z = (
                        np.max(zs[z_key][:, :, j])
                        if is_maximization
                        else np.min(zs[z_key][:, :, j])
                    )
                    max_x, max_y = np.unravel_index(
                        np.argmax(zs[z_key][:, :, j]), zs[z_key][:, :, j].shape
                    )
                    ax.scatter(
                        xs[x_key][max_x],
                        ys[y_key][max_y],
                        max_z,
                        color="red",
                        zorder=3.5,
                    )
                    ax.text(
                        xs[x_key][max_x],
                        ys[y_key][max_y],
                        max_z,
                        f"({xs[x_key][max_x]}, {ys[y_key][max_y]}, {max_z:.2f})",
                    )

            x_label = x_key.split("_")[0] if ignore_suffix_x else x_key
            y_label = y_key.split("_")[0] if ignore_suffix_y else y_key
            ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
            ax.set_ylabel(y_label, fontsize=12, fontweight="bold")

            if labels is not None and len(labels) > 0:
                ax.set_zlabel(labels[i], fontsize=12, fontweight="bold")
        else:
            ax = fig.add_subplot(n_rows, n_cols, i + 1)

            for j in range(ys[y_key].shape[-1]):
                label = curve_labels[i][j] if curve_labels else None
                ax.plot(xs[x_key], ys[y_key][:, j], label=label, linewidth=3, alpha=0.7)

                if is_maximization is not None:
                    max_y = (
                        np.max(ys[y_key][:, j])
                        if is_maximization
                        else np.min(ys[y_key][:, j])
                    )
                    max_x = (
                        np.argmax(ys[y_key][:, j])
                        if is_maximization
                        else np.argmin(ys[y_key][:, j])
                    )
                    ax.scatter(xs[x_key][max_x], max_y, color="red", zorder=3)
                    ax.text(
                        xs[x_key][max_x],
                        max_y,
                        f"({xs[x_key][max_x]}, {max_y:.2f})",
                        fontsize=12,
                        fontweight="bold",
                    )

            x_label = x_key.split("_")[0] if ignore_suffix_x else x_key
            ax.set_xlabel(x_label, fontsize=12, fontweight="bold")

            if labels is not None and len(labels) > 0:
                ax.set_ylabel(labels[i], fontsize=12, fontweight="bold")

        if titles is not None and len(titles) > 0:
            ax.set_title(titles[i], fontsize=12, fontweight="bold")

        if xlim is not None:
            ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)

        if curve_labels is not None:
            ax.legend()

    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath)

    plt.show()
