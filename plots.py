import bqplot as bq
import ipywidgets as widgets
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from jax import grad, jit, random, vmap


def make_fig():
    fig = plt.figure()
    fig.canvas.toolbar_visible = False
    return fig


def plot_surface(U, V, Z, gradnorm, fig, ax, alpha=0.3, cmap=plt.cm.YlGn):
    scamap = plt.cm.ScalarMappable(cmap=cmap)
    fcolors = scamap.to_rgba(gradnorm)

    surf = ax.plot_surface(
        U,
        V,
        Z,
        color="lightblue",
        facecolors=fcolors,
        cmap=cmap,
        alpha=alpha,
    )

    clb = fig.colorbar(scamap)
    clb.ax.set_title("Norme du gradient")

    cset = ax.contour(U, V, Z, zdir="z", offset=Z.min() - 1.0, cmap=plt.cm.YlGn)
    ax.set_zlim(Z.min() - 1.0)


def plot_quiver(U, V, gradx, grady, gradnorm, ax, fig, cmap=plt.cm.YlGn):
    q = ax.quiver(U, V, gradx, grady, gradnorm, cmap=cmap)
    clb = fig.colorbar(q)
    clb.ax.set_title("Norme du gradient")
    return q


def gradient_descent(function, init, max_iter=10, lr=1.0):
    grad_fun = jit(grad(function))
    memo = []
    x = init
    for i in range(max_iter):
        memo.append(x)
        new_x = x - lr * grad_fun(x)
        x = new_x
    return jnp.asarray(memo)


def plot_gradient_descent(params, ax, alpha=1.0):
    scatter = ax.scatter(params[:, 0], params[:, 1], c="darkblue", alpha=alpha)
    annots = []
    for i in range(len(params)):
        annot = ax.annotate(
            "",
            xy=params[i + 1],
            xytext=params[i],
            arrowprops={"arrowstyle": "->", "color": "r", "lw": 2, "alpha": alpha},
            va="center",
            ha="center",
        )
        annots.append(annot)
    return scatter, annots


def plot_train(apply_fn, initial_params, x, y, memo, params):
    plt.ioff()
    fig_loss = make_fig()
    plt.ion()

    plt.plot(memo["loss"])
    plt.ylim(0)
    plt.title("Évolution du coût")
    plt.xlabel("Itérations")
    print(f"Coût initial: {memo['loss'][0]:.2f}")
    print(f"Coût final: {memo['loss'][-1]:.2f}")

    plt.ioff()
    fig_init = make_fig()
    plt.ion()

    plt.scatter(x, apply_fn(initial_params, x), c="darkblue", label="predicted")
    plt.scatter(x, y, c="red", label="actual")
    plt.legend()
    plt.title("Prédiction initiale")
    plt.xlabel("Feature")
    plt.ylabel("Cible")

    plt.ioff()
    fig_end = make_fig()
    plt.ion()

    plt.scatter(x, apply_fn(params, x), c="darkblue", label="predicted")
    plt.scatter(x, y, c="red", label="actual")
    plt.legend()
    plt.title("Prédiction finale")
    plt.xlabel("Feature")
    plt.ylabel("Cible")

    display(
        widgets.VBox([widgets.HBox([fig_init.canvas, fig_end.canvas]), fig_loss.canvas])
    )


def make_pred_bqplot():
    x_sc = bq.LinearScale()
    y_sc = bq.LinearScale()

    ax_x = bq.Axis(label="Feature", scale=x_sc)  # , tick_format="0.0f"
    ax_y = bq.Axis(
        label="Target", scale=y_sc, orientation="vertical"  # , tick_format="0.0e"
    )

    line = bq.Lines(
        x=[0],
        y=[0],
        scales={"x": x_sc, "y": y_sc},
        colors=["darkblue"],
        opacities=[1],
    )

    scatter = bq.Scatter(
        x=[0],
        y=[0],
        scales={"x": x_sc, "y": y_sc},
        colors=["red"],
        opacities=[1],
    )
    out_plot = bq.Figure(
        axes=[ax_x, ax_y],
        marks=[line, scatter],
        #     interaction=interval_selector,
        # animation_duration=100,
    )

    out_plot.legend_style = {"stroke-width": 0}
    out_plot.layout.width = "flex"
    return {k: v for k, v in locals().items()}
