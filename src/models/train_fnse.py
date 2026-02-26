import jax
import optax
from src.models.loss import denoising_score_matching_loss

## algorithm 1, step 5-7
def update_step(params, opt_state, key, theta_batch, x_t_batch, x_next_batch):
    
    grad_fn = jax.value_and_grad(denoising_score_matching_loss)
    loss, grads = grad_fn(params, score_net, key, theta, x_t, x_next, sde)

    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss



def sample_transitions(key, batch_size, simulator):
    k1, k2, k3 = jax.random.split(key, 3)

    # sample theta ~ p(\theta) = uniform(-5,5)
    thetas = jax.random.uniform(k1, (batch_size, 2), minval=-5.0, maxval=5.0)

    # sample x_t ~ p_tilde(x_t) = normal
    x_ts = jax.random.normal(k2, (batch_size, 2))

    step_vmap = jax.vmap(simulator, in_axes=(0, 0, 0))
    keys = jax.random.split(k3, batch_size)
    x_nexts = step_vmap(x_ts, thetas, keys)

    return thetas, x_ts, x_nexts


## algorithm 1, step 3-8
for step in range(num_steps):
    key, subkey = jax.random.split(key)

    theta_b, x_t_b, x_next_b = sample_transitions(subkey, batch_size)

    params, opt_state, loss_val = update_step(
        params, opt_state, subkey, theta_b, x_t_b, x_next_b
    )

    if step % 100 == 0:
        print(f"Step {step}: Loss = {loss_val:.4f}")