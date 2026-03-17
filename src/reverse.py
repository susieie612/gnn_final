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
        score_net: InferScoreNet, LocalScoreNet
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
        x_pair: (x_1,t, x_1,t+1, x_2,t, x_2,t+1....)  (shape: (2 * d_x))
        """

        theta = state.position # (d_tehta)
        t = state.time 
        dt = a_new - t # dt is negative

        # to match the shape of LocalScoreNet (B, 2*d_x)
        x_in = x_pair[jnp.newaxis, ...] # (1,2*d_x)
        theta_in = theta[jnp.newaxis, ...] # (1, d_theta)
        a_in = jnp.atleast_1d(t) # (1,)
        # print(f'[Reverse, EulerMaruyama] x_in shape: {x_in.shape}')

        score = self.score_net.apply( # trained ScoreMLP
            {'params': self.params},
            x_in, 
            theta_in, 
            a_in
            )
        score = jnp.squeeze(score, axis=0) # (d_theta, )
        score = jnp.clip(score, -50, 50)

        # # noise prediction version
        # noise_pred = self.score_net.apply(
        #     {'params': self.params},
        #     x_in, 
        #     theta_in, 
        #     a_in
        #     )
        # noise_pred = jnp.squeeze(noise_pred, axis=0)
        
        # # transfer noise into score 
        # std_t = self.sde.std(t)
        # score = -noise_pred / (std_t + 1e-8)
        # score = jnp.clip(score, -1e3, 1e3)

        # Update with Euler-Maruyama
        beta_t = self.sde.beta_min + t * (self.sde.beta_max - self.sde.beta_min)
        drift = (-0.5 * beta_t * theta - beta_t * score) * dt
        diffusion = jnp.sqrt(beta_t) * jnp.sqrt(jnp.abs(dt)) * jax.random.normal(key, theta.shape)
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
    def sample_conditional(self, key, x_pair, a_start, init_pos):
        """
        sample theta trhough the reverse process starting from a_start to 0
        to be called inside estimate_local_precision of GAUSSScoreFn 
        """
        # print(f"--- [REVERSE DEBUG] sample x_pair shape: {x_pair.shape} ---")
        # initialize
        sampling_grid = self.time_grid[::-1]  # reverse the input time grid

        # std_a = self.sde.std(a_start)
        # k_init, k_loop = jax.random.split(key)
        # initial_position = std_a * jax.random.normal(k_init, self.theta_shape)

        # initalize the kernel, which is Euler-Maruyama
        state = self.kernel.init(init_pos, x_pair, a_start)

        def body_fn(carry, a_new):
            k, current_state = carry
            k, sk = jax.random.split(k)
            
            # a step of reverse process
            next_state = jax.lax.cond(
                a_new < a_start, # for case where a \= 1.0, then skip all time step that is larger than starting a
                lambda s: self.kernel.step(sk, s, a_new, x_pair), # if a_new < a_start
                lambda s: s._replace(time=a_new),  # if a_new > a_start, only update time
                current_state
            )

            # if self.transform_state is not None:
            #     next_state = self.transform_state(next_state)
            
            return (k, next_state), None
        
        # loop over entire reverse process
        (_, final_state), _ = jax.lax.scan(
            body_fn, 
            (key, state), 
            sampling_grid
        )

        return final_state.position

    @partial(jax.jit, static_argnums=(0, ))
    def sample(self, key, x_pair, a_start):
        """start from new noise for sampling"""
        std_a = self.sde.std(a_start)
        k_init, k_loop = jax.random.split(key)
        initial_position = std_a * jax.random.normal(k_init, self.theta_shape)
        return self.sample_conditional(k_loop, x_pair, a_start, initial_position)