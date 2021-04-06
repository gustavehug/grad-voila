from itertools import chain

import bqplot as bqplt
from jax.experimental import stax
from jax.experimental.stax import Dense, Dropout, Flatten, Identity, Relu, Tanh
from jax.random import PRNGKey

from gradient import *
from plots import *


# create artificial regression dataset
def get_data(N=100, D_X=3, sigma_obs=0.05, N_test=100, seed=0):
    D_Y = 1  # create 1d outputs
    np.random.seed(seed)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y *= 10
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, Y, X_test


def split(a, key, test_ratio=0.3):
    shuffled = random.permutation(key, a)
    on = jnp.int32(jnp.floor(test_ratio * a.shape[0]))
    return shuffled[on:], shuffled[:on]


def descend_and_update(loss_xy):
    global params, memo
    params, memo = train_opt(loss_xy, n_train, init_params, lr, momentum)
    #     memo_params = jnp.dstack(jax.tree_map(jnp.ravel, memo["params"]))[0]
    #     memo_params = jnp.vstack([jnp.dstack(initial_params)[0], memo_params])

    #     lines.set_data(memo_params[:, 0], memo_params[:, 1])
    loss_plot["line"].y = memo["loss"]

    update_step({"new": step_slider.value})


def update_step(change):
    step = change["new"]
    current_params = jax.tree_map(lambda a: a[step], memo["params"])
    loss_plot["scatter"].x = [step]
    loss_plot["scatter"].y = [memo["loss"][step]]

    pred_plot["line"].y = nn_apply_fn(current_params, xlin)

    label.value = f'Loss: {memo["loss"][step]:.3f}'


def on_slider_update(change):
    global lr, momentum
    if change["owner"].description == "log(lr)":
        lr = 10 ** change["new"]
    #  change["owner"].description = "LR : " + str(round(lr, 3))

    elif change["owner"].description == "momentum":
        momentum = change["new"]

    descend_and_update(nn_loss_xy)


X, Y, X_test = get_data(seed=5)
key = PRNGKey(0)
x_tr, x_te = split(X, key)
x_tr = x_tr[:, [1]]
x_te = x_te[:, [1]]
y_tr, y_te = split(Y, key)

n_layers = 3
n_neurons = 3

nn_init_fn, nn_apply_fn = stax.serial(
    *chain(*[(Tanh, Dense(n_neurons)) for _ in range(n_layers)]),
    Dense(1),
)

out_shape, init_params = nn_init_fn(PRNGKey(9), x_tr.shape[1:])

n_train = 400
lr = 0.125
momentum = 0.9
nn_loss = partial(make_loss, nn_apply_fn)
nn_loss_xy = partial(nn_loss, x=x_tr, y=y_tr)
params, memo = train_opt(nn_loss_xy, n_train, init_params, lr, momentum)

x = x_te
y = y_te
xlin = jnp.linspace(x.min() - 1, x.max() + 1, 50).reshape(-1, 1)

pred_plot = make_pred_bqplot()
pred_plot["line"].x = xlin
pred_plot["line"].y = nn_apply_fn(params, xlin)

pred_plot["y_sc"].min = float(y.min()) - 2.0
pred_plot["y_sc"].max = float(y.max()) + 2.0

pred_plot["scatter"].x = x.ravel()
pred_plot["scatter"].y = y.ravel()


def make_loss_bqplot():
    x_sc = bq.LinearScale()
    y_sc = bq.LinearScale()

    ax_x = bq.Axis(label="Iteration", scale=x_sc)  # , tick_format="0.0f"
    ax_y = bq.Axis(
        label="Loss", scale=y_sc, orientation="vertical"  # , tick_format="0.0e"
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
    return {k: v for k, v in locals().items()}


step = 0

loss_plot = make_loss_bqplot()

loss_plot["line"].x = jnp.arange(n_train)
loss_plot["line"].y = memo["loss"]

loss_plot["scatter"].x = [step]
loss_plot["scatter"].y = [memo["loss"][step]]

out = widgets.Output()
label = widgets.Label(value=f'{memo["loss"][-1]:.3f}')


lr_slider = widgets.FloatSlider(min=-3.0, max=-0.7, value=-1.8, description="log(lr)")
lr_slider.observe(on_slider_update, "value")

momentum_slider = widgets.FloatSlider(
    min=0.0, max=0.95, step=0.05, value=momentum, description="momentum"
)
momentum_slider.observe(on_slider_update, "value")

n_train_val = memo["loss"].shape[0]
step_size = n_train_val / n_train_val

step_slider = widgets.IntSlider(
    min=0, max=n_train_val, value=0, step=step_size, description="step"
)
step_slider.observe(update_step, "value")

play = widgets.Play(
    value=n_train_val,
    min=0,
    max=n_train_val,
    step=step_size,
    interval=30,
    description="Descend",
    disabled=False,
)
widgets.jslink((play, "value"), (step_slider, "value"))

layout = widgets.Layout(grid_template_columns="1fr 1fr")

explo = widgets.GridBox(
    children=[
        loss_plot["out_plot"],
        pred_plot["out_plot"],
        widgets.VBox(children=[lr_slider, momentum_slider]),
        widgets.VBox(children=[widgets.HBox(children=[play, step_slider]), label]),
    ],
    layout=layout,
)
