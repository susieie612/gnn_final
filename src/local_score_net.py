import flax.linen as nn
import jax.numpy as jnp


class LocalScoreNet(nn.Module):
    config: dict # config['model'] in main.py

    @nn.compact
    def __call__(self,
                x_t, # total observation (B, T, dim(x))
                theta,  # parameters (B, dim(theta))
                a # diffusion time (B, )
                ):
        """
        splits x_t into local observations and estimates the score function of piecewise posterior distribution p(theta | x_t, a)
        returns an array of score estimates for each local observation 
        """

        B, T, d_x = x_t.shape
        h_dim = self.config['hidden_dim']
        d_teta = self.config['d_teta']

        # split x_t into local observations
        x_local = jnp.concatenate([x_t[:, :-1, :], x_t[:, 1:, :]], axis=-1) # (B, T-1, d_x * 2)
        
        # input layer
        x_emb = nn.Dense(h_dim)(x_local)
        theta_emb = nn.Dense(h_dim)(theta[:, jnp.newaxis, :]) # (B, 1, h_dim)

        # a embedding
        a_proj = nn.Dense(h_dim)(a[:, jnp.newaxis])
        a_emb = nn.relu(a_proj)
        a_emb = nn.Dense(h_dim)(a_emb)[:, jnp.newaxis, :] # (B, 1, h_dim)

        # output of input layer
        h = x_emb + theta_emb + a_emb


        for _ in range(3):
                res = h
                h = nn.Dense(h_dim)(h)
                h = nn.relu(h)
                h = h + res + a_emb
        # return the array of score estimates for each local observation
        return nn.Dense(d_teta)(h) # (B, T-1, d_theta)



