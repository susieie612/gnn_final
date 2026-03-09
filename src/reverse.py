import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple

class EMState(NamedTuple):
    """To save reverse diffusion process"""
    position: jnp.ndarray
    time: jnp.ndarray


class EulerMaruyama: # initilize with score function and sde
    def __init__(self, score_net, params, sde): 
        """
        Args:
        score_net: LocalScoreNet
        params: leearned network parameters
        sde: SDE
        """
        self.score_net = score_net
        self.params = params
        self.sde = sde

    def init(self, theta_0, x_pair, a_start):
        """Call in Diffuser to iniitalize the state"""
        return EMState(position=theta_0, time=a_start)

    def step(self, key, state, a_new, x_pair):
        """
        Single reverse diffusion step
        Args:
        state: current (theta, time)
        a_new: next time step value
        x_pair: (x_t, x_t+1) 
        """

        theta = state.position
        t = state.time 
        dt = a_new - t

        # to match the shape of LocalScoreNet (B, T, d_x)
        x_in = x_pair[jnp.newaxis, jnp.newaxis, :]
        theta_in = theta[jnp.newaxis, ...]
        t_in = t[jnp.newaxis, ...]

        # calculate the local score
        score = self.score_net.apply(self.params, x_in, theta_in, t_in)
        score = jnp.squeeze(score)

        # Update with Euler-Maruyama
        g = self.sde.std(t) 

        drift = (g**2) * score * dt
        diffusion = g * jnp.sqrt(jnp.abs(dt)) * jax.random.normal(key, theta.shape)

        theta_next = theta + drift + diffusion

        return EMState(position=theta_next, time=a_new)


class Diffuser:
    def __init__(self, kernel, time_grid, theta_shape, sde, transform_state=None): 
        """
        sample from local posterior p(theta|theta_a, x^t,t+1) 
        """
        self.kernel = kernel # Euler-Maruyama
        self.time_grid = time_grid
        self.theta_shape = theta_shape
        self.sde = sde 
        self.transform_state = transform_state

    @partial(jax.jit, static_argnums=(0, ))
    def sample(self, key, x_pair, a_start):
        """
        sample theta trhough the reverse process starting from a_start to 0
        to be called inside estimate_local_precision of GAUSSScoreFn 
        """
        # initialize
        sampling_grid = self.grid[self.grid <= a_start][::-1] # 

        std_a = self.sde.std(a_start)
        initial_position = std_a * jax.random.normal(key, self.theta_shape)

        # initalize the kernel
        state = self.kernel.init(initial_position, x_pair, a_start)

        def body_fn(carry, a_new):
            k, current_state = carry
            k, sk = jax.random.split(k)
            
            # a step of reverse process
            next_state = self.kernel.step(sk, current_state, a_new, x_pair)

            if self.transform_state is not None:
                next_state = self.transform_state(next_state)
            
            return (k, next_state), None
        
        # loop over entire reverse process
        (_, final_state), _ = jax.lax.scan(
            body_fn, 
            (key, state), 
            sampling_grid[1:]
        )

        return final_state.position