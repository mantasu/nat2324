import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable


@plt.style.context("dark_background")
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
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    plt.tight_layout()

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
        plt.savefig(filepath, format="jpg", dpi=300)
    
    # Show plot
    plt.show()


# @plt.style.context("bmh")
def visualize_optimization_experiments(
    xs: dict[str, np.ndarray],
    ys: dict[str, np.ndarray],
    zs: dict[str, np.ndarray] | None = None,
    labels: str | list[str | list[str]] | None = None,
    max_cols: int = 3,
    is_maximization: bool | None = None,
    filepath: str | None = None,
    scale: tuple[int, int] = (4, 4)
):
    if labels is not None and (isinstance(labels, str) or isinstance(labels[0], str)):
        # Init new labels
        new_labels = []

        for key in ys.keys():
            if isinstance(labels, list):
                # Assume dims of same size
                new_labels.append(labels)
            else:
                # Compute the dimensionality, use that many labels
                num_labels = 1 if ys[key].ndim == 1 else len(ys[key])
                new_labels.append([labels] * num_labels)
        
        # Convert to a list of lists
        labels = new_labels
    
    n_cols = min(len(xs), max_cols)
    n_rows = len(xs) // n_cols + (len(xs) % n_cols > 0)

    fig = plt.figure(figsize=(n_cols * scale[0], n_rows * scale[1]), layout="constrained")
    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # Adjust subplot parameters
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2, left=-0.2, right=0.8, top=0.8)
    
    for i, (x_key, y_key) in enumerate(zip(xs.keys(), ys.keys())):
        if zs is not None:
            # plt.cla()  # clear current axes
            ax = fig.add_subplot(n_rows, n_cols, i+1, projection="3d")
            ax.view_init(elev=30., azim=-150.)
            ax.set_box_aspect(aspect=None, zoom=0.85)

            X, Y = np.meshgrid(xs[x_key], ys[y_key])
            z_key = list(zs.keys())[i]

            for j in range(zs[z_key].shape[-1]):
                Z = np.moveaxis(zs[z_key][:,:,j], 0, 1)
                ax.plot_surface(X, Y, Z, label=labels[i][j] if labels else None, alpha=0.7, cmap="viridis", cstride=1, rstride=1)

                if is_maximization is not None:
                    max_z = np.max(zs[z_key][:,:,j]) if is_maximization else np.min(zs[z_key][:,:,j])
                    max_x, max_y = np.unravel_index(np.argmax(zs[z_key][:,:,j]), zs[z_key][:,:,j].shape)
                    ax.scatter(xs[x_key][max_x], ys[y_key][max_y], max_z, color='red', zorder=3.5)
                    ax.text(xs[x_key][max_x], ys[y_key][max_y], max_z, f'({xs[x_key][max_x]}, {ys[y_key][max_y]}, {max_z})')
        else:
            ax = fig.add_subplot(n_rows, n_cols, i+1)

            for j in range(ys[y_key].shape[-1]):
                label = labels[i][j] if labels else None
                ax.plot(xs[x_key], ys[y_key][:, j], label=label, linewidth=3)

                if is_maximization is not None:
                    max_y = np.max(ys[y_key][:, j]) if is_maximization else np.min(ys[y_key][:, j])
                    max_x = np.argmax(ys[y_key][:, j]) if is_maximization else np.argmin(ys[y_key][:, j])
                    ax.scatter(xs[x_key][max_x], max_y, color='red', zorder=3)
                    ax.text(xs[x_key][max_x], max_y, f'({xs[x_key][max_x]}, {max_y})', fontsize=12, fontweight='bold')
        
        ax.set_xlabel(x_key, fontsize=12, fontweight="bold")
        ax.set_ylabel(y_key, fontsize=12, fontweight="bold")

        if zs is not None:
            zlabel = ax.set_zlabel(z_key, fontsize=12, fontweight="bold")
            # zlabel.set_position((-1,0))

        if labels is not None:
            ax.legend()
    
    if filepath:
        plt.savefig(filepath)
    
    plt.show()