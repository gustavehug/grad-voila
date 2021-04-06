from jax.random import PRNGKey

from gradient import *
from plots import *


def descend_and_update(loss_xy):
    global params, memo
    params, memo = train_opt(loss_xy, n_train, initial_params, lr, momentum)
    memo_params = jnp.dstack(jax.tree_map(jnp.ravel, memo["params"]))[0]
    memo_params = jnp.vstack([jnp.dstack(initial_params)[0], memo_params])

    lines.set_data(memo_params[:, 0], memo_params[:, 1])

    update_step({"new": step_slider.value})


def update_step(change):
    step = change["new"]
    current_params = jax.tree_map(lambda a: a[step], memo["params"])
    # focus.set_data(current_params[0][0], current_params[1])

    pred_plot["line"].y = sin_apply_fn(current_params, xlin)
    # sin_line.set_data(xlin, sin_apply_fn(current_params, xlin))

    label.value = f'Loss: {memo["loss"][step]:.3f}'


def on_click(event):
    global initial_params

    init_w, init_b = event.xdata, event.ydata

    if init_w is None:
        return

    initial_params = (jnp.asarray([init_w]), init_b)
    descend_and_update(sin_loss_xy)


def on_slider_update(change):
    global lr, momentum
    if change["owner"].description == "log(lr)":
        lr = 10 ** change["new"]
    #  change["owner"].description = "LR : " + str(round(lr, 3))

    elif change["owner"].description == "momentum":
        momentum = change["new"]

    descend_and_update(sin_loss_xy)


N = 100
P = 1

x, y, w_true, b_true = make_sine_regression(PRNGKey(2), 100, 1)

sin_loss = partial(make_loss, sin_apply_fn)
n_train = 30

initial_params = init_fn(PRNGKey(1), P)
params, memo = train(sin_loss, n_train, initial_params, x, y, lr=0.1)

sin_loss_xy = partial(sin_loss, x=x, y=y)

xmin, xmax = memo["params"][0].min(), memo["params"][0].max()
ymin, ymax = memo["params"][1].min(), memo["params"][1].max()

xvar = xmax - xmin
yvar = ymax - ymin

# xrange = xmin - xvar, xmax + xvar
# yrange = ymin - yvar, ymax + yvar
xrange = (-2, 2)
yrange = (-5, 5)


U, V, Z, pairs, gradvals, gradx, grady, gradnorm = make_gradient_field(
    sin_loss_xy, xrange=xrange, yrange=yrange, n_points=20
)

plt.ioff()
fig_grad, ax = plt.subplots()
plt.ion()
fig_grad.canvas.toolbar_visible = False
plot_quiver(U, V, gradx, grady, gradnorm, ax, fig_grad)
plt.xlabel("Weight")
plt.ylabel("Bias")

# plt.ioff()
# fig_grad = make_fig()
# plt.ion()

# ax = fig_grad.gca(projection="3d")
# plot_surface(U, V, Z, gradnorm, fig_grad, ax)
# plt.xlabel("Poids")
# plt.ylabel("Biais")
# ax.set_zlabel("Coût")
# plt.title("Coût en fonction des paramètres")

memo_params = jnp.dstack(jax.tree_map(jnp.ravel, memo["params"]))[0]
memo_params = jnp.vstack([jnp.dstack(initial_params)[0], memo_params])
# scatter, annots = plot_gradient_descent(memo_params, ax=ax)
(lines,) = ax.plot(
    memo_params[:, 0],
    memo_params[:, 1],
    #         zs=-1.0,
    #     vmap(sin_loss_xy)(memo["params"]),
    marker=".",
)
plt.title("Gradient descent\nand vector field")
# (focus,) = ax.plot(memo_params[0, 0], memo_params[0, 1], c="darkblue", marker=".")
plt.xlim(*xrange)
plt.ylim(*yrange)

xlin = jnp.linspace(x.min(), x.max(), 50).reshape(-1, 1)

pred_plot = make_pred_bqplot()
pred_plot["line"].x = xlin
pred_plot["line"].y = sin_apply_fn(params, xlin)

pred_plot["scatter"].x = x.ravel()
pred_plot["scatter"].y = y

out = widgets.Output()
label = widgets.Label(value=f'{memo["loss"][-1]:.3f}')

lr = 0.1
momentum = 0.7

lr_slider = widgets.FloatSlider(min=-3.0, max=-0.5, value=-1.0, description="log(lr)")
lr_slider.observe(on_slider_update, "value")

momentum_slider = widgets.FloatSlider(
    min=0.0, max=1.0, value=momentum, description="momentum"
)
momentum_slider.observe(on_slider_update, "value")

n_train_val = memo["loss"].shape[0]
step_size = n_train_val // n_train_val
step_slider = widgets.IntSlider(
    min=0, max=n_train_val, step=step_size, value=0, description="step"
)
step_slider.observe(update_step, "value")

play = widgets.Play(
    value=n_train_val,
    min=0,
    max=n_train_val,
    step=step_size,
    interval=50,
    description="Descend",
    disabled=False,
)
widgets.jslink((play, "value"), (step_slider, "value"))

cid = fig_grad.canvas.mpl_connect("button_press_event", on_click)

fig_grad.canvas.blit()

layout = widgets.Layout(grid_template_columns="1fr 1fr")

explo = widgets.GridBox(
    children=[
        fig_grad.canvas,
        pred_plot["out_plot"],
        widgets.VBox(children=[lr_slider, momentum_slider]),
        widgets.VBox(children=[widgets.HBox(children=[play, step_slider]), label]),
    ],
    layout=layout,
)