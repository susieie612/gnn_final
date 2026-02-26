import jax
import jax.numpy as jnp

def step_mixture_rw(x_t, theta, key):
    """
    Updates a step of Gaussian Mixture RW (2D)
    x^{t+1} = x_t + u * theta + epsilon
    """

    k1, k2 = jax.random.split(key) # fix the random states

    # u ~ Unif({-1,1})
    u = jax.random.choice(k1, jnp.array([-1.0, 1.0]))

    # espsilon ~ N(0, I)
    epsilon = jax.random.normal(k2, shape=x_t.shape)

    x_next = x_t + u * theta + epsilon

    return x_next


def mixture_rw_trajectory(theta, x_0, T, key):
    
    def scan_fn(x_prev, k):
        x_next = step_mixture_rw(x_prev, theta, k)
        return x_next, x_next # carry(x_prev in next step), output(to be added to trajectory)
    
    keys = jax.random.split(key, T)
    x_T, trajectory = jax.lax.scan(scan_fn, x_0, keys)

    return jnp.concatenate([x_0[None, :], trajectory], axis=0)
