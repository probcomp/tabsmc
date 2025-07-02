import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool
from functools import partial


# @partial(jax.jit, static_argnames=("batch_size",))
def js(x: Bool[Array, "n k"], y: Bool[Array, "m k"], batch_size: int = 1000) -> Array:
    k = x.shape[1]
    col_combinations = jnp.array([
        [i, j] for i in range(k) for j in range(i + 1, k)
    ])

    x_indexed = jax.vmap(lambda c: x.take(c, axis=1))(col_combinations)
    y_indexed = jax.vmap(lambda c: y.take(c, axis=1))(col_combinations)

    # return jax.lax.map(batch_js_bivariate, (x_indexed, y_indexed), batch_size=batch_size)
    return jax.vmap(js_bivariate, in_axes=(0, 0))(x_indexed, y_indexed)

def batch_js_bivariate(args):
    x, y = args
    return js_bivariate(x, y)

def js_bivariate(x: Bool[Array, "n 2"], y: Bool[Array, "m 2"]) -> Array:
    vals = jnp.array([
        [True, True],
        [True, False],
        [False, True],
        [False, False],
    ])

    x_vals = jax.vmap(jax.vmap(jnp.array_equal, in_axes=(0, None)), in_axes=(None, 0))(x, vals)
    x_vals = jnp.mean(x_vals, axis=1)
    y_vals = jax.vmap(jax.vmap(jnp.array_equal, in_axes=(0, None)), in_axes=(None, 0))(y, vals)
    y_vals = jnp.mean(y_vals, axis=1)

    m_vals = (x_vals + y_vals) / 2

    kl_x_m = jnp.nansum(x_vals * jnp.log(x_vals / m_vals)) 
    kl_y_m = jnp.nansum(y_vals * jnp.log(y_vals / m_vals))

    return (kl_x_m + kl_y_m) / 2
