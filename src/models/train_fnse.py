import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial

from src.models.loss import denoising_score_matching_loss

## define SDE
class SDE:
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T_min = 1e-4
        self.T_max = 1.0

    def std(self, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return jnp.sqrt(1.0 - jnp.exp(2.0 * log_mean_coeff))

    def mean(self, t, theta):
        log_mean = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return jnp.exp(log_mean) * theta
    

def train_fnse(model, sampler, num_steps=5000, batch_size=128):
    sde = SDE()
    optimizer = optax.adam(learning_rate=1e-3)

    # initialize model parameters
    key = jax.random.PRNGKey(0)
    init_data = (jnp.ones((1, 2)), jnp.ones((1, 2)), jnp.ones((1, 2)), jnp.ones((1, 1)))
    params = model.init(key, *init_data)['params']
    opt_state = optimizer.init(params)

    # algorithm 1, Line 5-7
    @jax.jit
    def update_step(params, opt_state, key, theta_b, xt_b, x_next_b):
        
        loss, grads = jax.value_and_grad(denoising_score_matching_loss)(params, model, key, theta_b, xt_b, x_next_b, sde)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    # training loop
    for step in range(num_steps):
        key, subkey = jax.random.split(key, 3)

        # algorithm 1, Line 4: sample a batch of data
        theta_b, xt_b, x_next_b = sampler.sample(subkey, batch_size)
        params, opt_state, loss = update_step(params, opt_state, key, theta_b, xt_b, x_next_b)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

    return params
