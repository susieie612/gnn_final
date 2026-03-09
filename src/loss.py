import jax
import jax.numpy as jnp


def denoising_score_matching_loss(params, score_net, key, theta, x_t, x_next, sde):
    """
    Eq. 4 of the paper (Denoising score matching loss)
    """

    k1, k2 =jax.random.split(key)

    # sample diffusion time from uniform distribution
    # TODO: later could change the distribution of a to focus training on certain level of noise
    a = jax.random.uniform(k1, (theta.shape[0],1), minval = sde.T_min, maxval = sde.T_max) 

    eps = jax.random.normal(k2, shape = theta.shape)
    std_a = sde.std(a)
    mean_a = sde.mean(a, theta)
    theta_perturbed = mean_a + std_a * eps

    score_pred = score_net.apply(params, theta_perturbed, x_t, x_next, a)

    score_target = -eps / std_a

    loss = jnp.mean(jnp.sum((score_pred - score_target)**2, axis=-1))
    return loss




