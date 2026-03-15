import jax
import jax.numpy as jnp


def denoising_score_matching_loss(params, model, key, theta, x_train, sde):
    """
    Eq. 4 of the paper (Denoising score matching loss)
    """

    k1, k2 =jax.random.split(key)

    # sample diffusion time from uniform distribution
    # TODO: later could change the distribution of a to focus training on certain level of noise
    a = jax.random.uniform(k1, (theta.shape[0],), minval = sde.T_min, maxval = sde.T_max) 

    # add the sampled noise to theta
    eps = jax.random.normal(k2, shape = theta.shape)
    std_a = sde.std(a)[:, jnp.newaxis] # sde defined in main.py
    mean_a = sde.mean(a, theta)
    theta_perturbed = mean_a + std_a * eps

    # concatenate to pass all x pairs seperately parallel through the ScoreMLP
    # x_input = jnp.concatenate([x_t, x_next], axis=-1) # (B, 2*d_x)
    x_input = x_train.reshape(x_train.shape[0], -1)
    
    score_pred = model.apply( # ScoreMLP
        {'params': params},
         x_input, # (B, 2*d_x)
         theta_perturbed, # (B, d_theta)
         a[:, jnp.newaxis] # (B, 1)
         )

    score_target = -eps / (std_a + 1e-8) # true noise vector
    score_target = score_target[:, jnp.newaxis, :]

    loss = jnp.mean(jnp.sum((score_pred - score_target)**2, axis=-1))

    return loss




