from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from jax.experimental import optimizers

# @partial(jit, static_argnums=(0, 3, 4))
def make_gradient_field(
    function, xrange=(-1, 2), yrange=(-1, 2), n_points=30, shape=(2, 1)
):
    W = jnp.linspace(*xrange, n_points)
    B = jnp.linspace(*yrange, n_points)
    U, V = jnp.meshgrid(W, B)
    pairs = jnp.dstack([U, V]).reshape(-1, *shape)

    vectorized_fun = jit(vmap(function))
    Z = vectorized_fun(pairs).reshape(n_points, n_points)

    grad_fun = jit(vmap(grad(function)))
    gradvals = grad_fun(pairs)

    gradx = gradvals[:, 0].reshape(n_points, n_points)
    grady = gradvals[:, 1].reshape(n_points, n_points)

    gradnorm = jnp.sqrt(gradx ** 2 + grady ** 2)

    return U, V, Z, pairs, gradvals, gradx, grady, gradnorm


@partial(jit, static_argnums=(0, 1))
def train_opt(loss_fn_xy, size, initial_params, lr, momentum):
    opt_init, opt_update, get_params = optimizers.momentum(lr, momentum)

    def step(step, opt_state):
        loss, grads = jax.value_and_grad(loss_fn_xy)(get_params(opt_state))
        opt_state = opt_update(step, grads, opt_state)
        return loss, opt_state

    def scan_fn(opt_state, i):
        loss, opt_state = step(i, opt_state)
        return opt_state, {"loss": loss, "params": get_params(opt_state)}

    def train(initial_params):
        init_opt_state = opt_init(initial_params)
        opt_state, memo = jax.lax.scan(scan_fn, init_opt_state, jnp.arange(size))
        return get_params(opt_state), memo

    return train(initial_params)


def make_sine_regression(key, n, p):
    key1, key2, key3 = random.split(key, 3)
    x = 2 * random.normal(key1, [n, p])
    w = random.normal(key2, [p])
    b = random.normal(key3)
    noise = 0.2 * random.normal(key3, [n])

    # actual data generation process
    y = jnp.sin(x @ w + b) + noise
    return x, y, w, b


def make_linear_regression(key, n, p):
    key1, key2, key3 = random.split(key, 3)
    x = random.normal(key1, [n, p])
    w = random.normal(key2, [p])
    b = random.normal(key3)
    noise = 0.5 * random.normal(key3, [n])

    # actual data generation process
    y = x @ w + b + noise
    return x, y, w, b


def init_fn(key, n_features):
    key1, key2 = random.split(key)
    w = random.normal(key1, [n_features])
    b = random.normal(key2)
    return w, b


def linear_apply_fn(params, x):
    w, b = params
    return x @ w + b


# just-in-time compilation, ignore function argument
@partial(jit, static_argnums=0)
def make_loss(apply_fn, params, x, y):
    return jnp.mean((apply_fn(params, x) - y) ** 2)


linear_loss = partial(make_loss, linear_apply_fn)
grad_linear_loss = grad(linear_loss)
# linear_loss_and_grad = jit(jax.value_and_grad(linear_loss))


@partial(jit, static_argnums=(0, 1))
def train(loss, size, initial_params, x, y, tolerance=0.005, lr=0.1):
    grad_loss = grad(loss)

    def scan_fn(params, _):
        current_loss = loss(params, x, y)
        current_grad = grad_loss(params, x, y)

        # gradient descent step
        params = jax.tree_multimap(
            lambda val, grd: val - lr * grd,
            params,
            current_grad,
        )
        #         return params, jnp.hstack([current_loss, *params, *current_grad])
        return params, {"loss": current_loss, "params": params}

    params, memo = jax.lax.scan(scan_fn, initial_params, jnp.arange(size))
    return params, memo


def sin_apply_fn(params, x):
    w, b = params
    out = jnp.sin(x @ w + b)
    return out
