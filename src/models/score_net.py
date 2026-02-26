import flax.linen as nn
import jax.numpy as jnp

class ScoreNet(nn.Module):
    hidden_dim: int = 128
    out_dim: int = 2

    @nn.compact

    def __call__(self, theta_a, x_t, x_next, a):

        inputs = jnp.concatenate([theta_a, x_t, x_next, a], axis = -1)

        x = nn.Dense(self.hidden_dim)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)

        return nn.Dense(self.out_dim)(x)