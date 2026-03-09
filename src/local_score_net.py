import flax.linen as nn
import jax.numpy as jnp


class LocalScoreNet(nn.Module):
    d_teta: int
    hidden_dim: int = 128

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
        d_theta = theta.shape[-1]

        # split x_t into local observations
        x_local = jnp.concatenate([x_t[:, :-1, :], x_t[:, 1:, :]], axis=-1) # (B, T-1, d_x * 2)
        
        # a embedding
        a_emb = nn.Dense(self.hidden_dim)(a.reshape(B, 1))
        a_emb = nn.relu(a_emb)
        a_emb = nn.Dense(self.hidden_dim)(a_emb) # (B, hidden_dim)

        # broadcast theta to (B,T-1,d_theta)
        # theta_in = jnp.repeat(theta[:, None, :], T-1, axis=1)
        # a_emb_in = jnp.repeat(a_emb[:, None, :], T-1, axis=1)
        theta_in = theta[:, jnp.newaxis, :]  # (B, 1, d_theta)
        a_emb_in = a_emb[:, jnp.newaxis, :] # (B, 1, hidden_dim)

        # input to the network
        h = jnp.concatenate([x_local, theta_in, a_emb_in], axis=-1)

        for _ in range(3):
                h = h + a_emb[:, jnp.newaxis, :] 
                h = h + nn.Dense(self.hidden_dim)(theta_in)
                
                h = nn.Dense(self.hidden_dim)(h)
                h = nn.relu(h)

        # return the array of score estimates for each local observation
        return nn.Dense(self.d_teta)(h) # (B, T-1, d_theta)



