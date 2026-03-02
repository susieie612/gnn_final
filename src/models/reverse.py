import jax
import jax.numpy as jnp


## temporary test to see if the score function works
def clip_transform(min_val, max_val):
    def transform(state):
        return jnp.clip(state, min_val, max_val)
    return transform

class EulerMaruyama: # initilize with score function and sde
    def __init__(self, score_fn, sde): 
        self.score_fn = score_fn
        self.sde = sde

    def step(self, key, theta, t, dt):
        score = self.score_fn(theta, t)
        g = self.sde.std(t)
        
        drift = (g**2) * score * dt
        diffusion = g * jnp.sqrt(dt) * jax.random.normal(key, theta.shape)
        
        theta_next = theta + drift + diffusion
        return theta_next

class Diffuser:
    def __init__(self, kernel, time_grid, shape, sde, transform_state=None): # sde 추가
        self.kernel = kernel
        self.time_grid = time_grid
        self.shape = shape
        self.sde = sde 
        self.transform_state = transform_state

    def simulate(self, key):
        k_init, k_loop = jax.random.split(key)
        # initialize theta with noise according to SDE
        theta = jax.random.normal(k_init, self.shape) * self.sde.std(self.time_grid[0])
        
        def body_fn(i, val):
            k, x = val
            t = self.time_grid[i]
            next_t = self.time_grid[i+1]
            dt = t - next_t
            
            curr_key, next_key = jax.random.split(k)
            x_next = self.kernel.step(curr_key, x, t, dt)
            
            if self.transform_state is not None:
                x_next = self.transform_state(x_next)
                
            return next_key, x_next

        _, final_theta = jax.lax.fori_loop(0, len(self.time_grid)-1, body_fn, (k_loop, theta))
        return final_theta