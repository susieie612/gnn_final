import jax
import jax.numpy as jnp
import optax
from src.simulations.simulators import LotkaVolterra, lv_prior_predictive_proposal
from src.local_score_net import LocalScoreNet
from src.loss import denoising_score_matching_loss
from src.reverse import Diffuser, EulerMaruyama
from src.composition_score_fn import GAUSSScoreFn

class SDE:
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min, self.beta_max = beta_min, beta_max
        self.T_min, self.T_max = 1e-4, 1.0

    def std(self, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return jnp.sqrt(1.0 - jnp.exp(2.0 * log_mean_coeff))

def marginal_prior_score(a, theta, sde, prior):
    # Denoising prior score p(theta_a)
    return prior.score(theta) 

# ALGORITHM 1: train local score
def train(key, model, sim, proposal_fn):
    sde = SDE()
    optimizer = optax.adamw(learning_rate=5e-4) 
    
    init_theta = jnp.ones((1, 4)) # 4 params for LV
    init_x = jnp.ones((1, 2, 2))  # (B, T, d_x)
    params = model.init(key, init_x, init_theta, jnp.ones((1,)))['params']
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, k, theta, x_batch):
        loss, grads = jax.value_and_grad(denoising_score_matching_loss)(
            params, model, k, theta, x_batch[:, 0], x_batch[:, 1], sde
        ) 
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    print("Starting algorithm 1" )
    for i in range(10000):
        key, k_theta, k_prop, k_sim = jax.random.split(key, 4)
        theta = sim.prior_sampler(k_theta, 128)
        _, x_t = proposal_fn(k_prop, 128)
        x_next = sim.transition(k_sim, x_t, theta)
        x_batch = jnp.stack([x_t, x_next], axis=1)

        params, opt_state, loss = step(params, opt_state, key, theta, x_batch)
        if i % 1000 == 0: print(f"Step {i}, Loss: {loss:.4f}")
    
    return params

# ALGORITHM 2: compose for noise injected time series
def infer(key, model, params, sim, x_obs):
    sde = SDE()
    grid = jnp.linspace(1e-3, 1.0, 100) 
    
    # local score using diffuser (Algorithm 2, line 5-6)
    kernel = EulerMaruyama(model, params, sde)
    diffuser = Diffuser(kernel, grid, (4,), sde)
    
    # composition using GAUSS
    gauss_fn = GAUSSScoreFn(
        score_net=model, params=params, sde=sde, prior=None, 
        diffuser=diffuser, marginal_prior_fn=lambda a, t: marginal_prior_score(a, t, sde, None)
    )

    # sample from final model
    final_sampler = Diffuser(EulerMaruyama(gauss_fn, params, sde), grid, (4,), sde)
    return jax.vmap(lambda k: final_sampler.sample(k, x_obs, 1.0))(jax.random.split(key, 100))


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    lv = LotkaVolterra() 
    proposal = lv_prior_predictive_proposal(lv, key) 
    model = LocalScoreNet(d_teta=4)

    # Alg 1
    trained_params = train(key, model, lv, proposal) 
    
    # Alg 2
    # TODO: generate ground truth data using generate trajectory --> x_obs_target
    samples = infer(key, model, trained_params, lv, x_obs_target)  # x_obs_target (T, 2)
    print(f"completed sampling from the posterior: {samples.shape}")