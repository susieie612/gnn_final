import flax.linen as nn
import jax.numpy as jnp


class ScoreMLP(nn.Module):
    """local score network for single transition (x_t, x_t+1) for algorithm 1"""
    h_dim: int # hidden_dimension
    d_theta: int 

    @nn.compact
    def __call__(self, x_pair, theta, a):
        # x_pair: (B, 2 * d_x) --> (x_1_t, x_1_t+1, x_2_t, x_2_t+1, ...., x_d_t, x_d_t+1)
        # theta: (B, d_theta) 
        # a: (B, 1)

        # match x dimension for inference: (B,%)
        
        # inputs embedding
        x_emb = nn.Dense(self.h_dim)(x_pair)
        theta_emb = nn.Dense(self.h_dim)(theta)
        a_emb = nn.Dense(self.h_dim)(nn.relu(nn.Dense(self.h_dim)(a)))

        h = x_emb + theta_emb + a_emb

        for _ in range(3):
            res = h
            h = nn.Dense(self.h_dim)(h)
            h = nn.relu(h)
            h = h + res + a_emb
            
        return nn.Dense(self.d_theta)(h) # d_theta

# src/local_score_net.py

class LocalScoreNet(nn.Module):
    config: dict

    def setup(self):
    
        self.mlp = ScoreMLP(
            h_dim=self.config['hidden_dim'], 
            d_theta=self.config['d_theta']
        )

    @nn.compact
    def __call__(self, x_t, theta, a):
        """
        입력 x_t를 (N, 2 * d_x)로 변환하여 MLP에 전달
        """
        # for inference
        if x_t.ndim == 3:
            B, T, d_x = x_t.shape
            # transition paris
            x_input = jnp.concatenate([x_t[:, :-1, :], x_t[:, 1:, :]], axis=-1)
            
            # broadcast theta & a
            theta_ext = jnp.tile(theta[:, jnp.newaxis, :], (1, T-1, 1))
            a_ext = jnp.tile(a.reshape(B, 1, -1), (1, T-1, 1))
            
            # (B * (T-1), 2 * d_x)
            out = self.mlp(
                x_input.reshape(-1, 2 * d_x), 
                theta_ext.reshape(-1, theta.shape[-1]), 
                a_ext.reshape(-1, a_ext.shape[-1])
            )
            return out.reshape(B, T-1, -1)

        # for training
        else:
            x_input = x_t.reshape(x_t.shape[0], -1) 
            a_input = a.reshape(x_t.shape[0], -1)
            return self.mlp(x_input, theta, a_input)


# class LocalScoreNet(nn.Module):
#     config: dict

#     def setup(self):
#         # set ScoreMLP as a module
#         # this allows to pass the params of LocalScoreNet to ScoreMLP
#         self.mlp = ScoreMLP(
#             h_dim=self.config['hidden_dim'], 
#             d_theta=self.config['d_theta']
#         )

#     @nn.compact
#     def __call__(self, x_t, theta, a):
#         """
#         Splits total time series into x pairs : x_t (B, T, d_x) -> (B, T-1, 2*d_x)
#         and pass to ScoreMLP
#         For algorithm 2
#         """
#         print(f"[LocalScoreNet] x_t shape: {x_t.shape}")

#         if x_t.ndim == 2:
#             # 학습 시에는 이미 (x_t, x_next)가 합쳐진 x_input(B, 2*d_x)이 들어온다고 가정
#             return self.mlp(x_t, theta, a)

#         B, T, d_x = x_t.shape # Batch, steps, x dimension
        
#         # split into single transition pairs
#         x_local = jnp.concatenate([x_t[:, :-1, :], x_t[:, 1:, :]], axis=-1) # (B, T-1, 2*d_x)
        
#         # # Prepare ScoreMLP, treating time step (= T-1) as "batch"
#         # mlp = ScoreMLP(h_dim=self.config['hidden_dim'], d_theta=self.config['d_theta'])
        
#         # braodcast theta & a to match the shape (B, T-1, ..)
#         theta_ext = jnp.tile(theta[:, jnp.newaxis, :], (1, T-1, 1))
#         a_ext = jnp.tile(a[:, jnp.newaxis, jnp.newaxis], (1, T-1, 1))
        
#         print(f'theta_ext shape :{theta_ext.shape}')
#         print(f'a_ext shape :{a_ext.shape}')

#         return self.mlp(x_local, theta_ext, a_ext) # (B, T-1, d_theta)

