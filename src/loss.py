import jax
import jax.numpy as jnp


def denoising_score_matching_loss(params, score_net, key, theta, x_t, x_next, sde):
    """
    Eq. 4 of the paper (Denoising score matching loss)
    """

    k1, k2 =jax.random.split(key)

    # sample diffusion time from uniform distribution
    # TODO: later could change the distribution of a to focus training on certain level of noise
    a = jax.random.uniform(k1, (theta.shape[0],), minval = sde.T_min, maxval = sde.T_max) 

    # add perturbation for sampled noise level
    eps = jax.random.normal(k2, shape = theta.shape)
    std_a = sde.std(a)[:, jnp.newaxis]
    mean_a = sde.mean(a, theta)
    theta_perturbed = mean_a + std_a * eps

    x_batch = jnp.stack([x_t, x_next], axis=1) # broadcasting for input to local_score_net

    score_pred = score_net.apply(
        {'params': params},
        x_batch, # ( (x_0, x_1), (x_1, x_2), ...)
         theta_perturbed,
         a
         )
    score_target = -eps / (std_a + 1e-8)
    score_target = score_target[:, jnp.newaxis, :]

    loss = jnp.mean(jnp.sum((score_pred - score_target)**2, axis=-1))
    return loss




