import jax
import jax.numpy as jnp
import optax
from src.simulators import LotkaVolterra, lv_prior_predictive_proposal
from src.local_score_net import LocalScoreNet, ScoreMLP
from src.loss import denoising_score_matching_loss
from src.reverse import Diffuser, EulerMaruyama
from src.composition_score_fn import GAUSSScoreFn
from src.config import config
from src.visualise import plot_posterior_samples


class SDE:
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min, self.beta_max = beta_min, beta_max
        self.T_min, self.T_max = 1e-4, 1.0
    
    def mean_coeff(self, t):
        # t: (batch,)
        coeff = jnp.exp(-0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min)
        return coeff

    def mean(self, t, x0):
        # t: (batch, ),  x0: (128, 4)
        return self.mean_coeff(t)[:, jnp.newaxis] * x0

    def std(self, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        var = 1.0 - jnp.exp(2.0 * log_mean_coeff)
        return jnp.sqrt(jnp.maximum(var, 1e-10)) # clip the variance

def marginal_prior_score(a, theta, sde):
    # Denoising prior score p(theta_a)
    return -theta

# ALGORITHM 1: train local score
def train(key, model, sim, proposal_fn, config):
    sde = SDE()

    # get values from config dictionary
    lr = config.get('learning_rate', 5e-4)
    batch_size = config.get('batch_size', 128)
    optimizer = optax.adamw(learning_rate=lr) 

    d_x = sim.d_x
    d_theta = sim.d_theta
    
    # TODO: later update 2nd dim to match for general simulations
    init_x = jnp.ones((1, 2, sim.d_x)) 
    init_theta = jnp.ones((1, sim.d_theta))
    init_a = jnp.ones((1,))

    # create the network
    params = model.init(key, init_x, init_theta, init_a)['params']
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, k, theta, x_t, x_next):
        x_train = jnp.stack([x_t, x_next], axis=1) # (B, 2, d_x)

        loss, grads = jax.value_and_grad(denoising_score_matching_loss)(
            params, model, k, theta, x_train, sde
        ) 
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    print("Starting algorithm 1" )
    for i in range(10000):
        key, k_theta, k_prop, k_sim = jax.random.split(key, 4)
        
        # theta sampling from prior distribution
        theta = sim.prior_sampler(k_theta, batch_size)
        # sample x_t from proposal distribution p(x_t)
        _, x_t = proposal_fn(k_prop, batch_size)
        # do 1-step transition to get x_next
        x_next = sim.transition(k_sim, x_t, theta) # (B, d_x)

        params, opt_state, loss = step(params, opt_state, key, theta, x_t, x_next)
        
        if i % 1000 == 0: 
            print(f"Step {i}, Loss: {loss:.4f}")
    
    return params

# ALGORITHM 2: compose for noise injected time series
def infer(key, model, params, sim, x_obs):
    sde = SDE()
    grid = jnp.linspace(1e-3, 1.0, 100) 
    
    # 1. Create a local sampler for precision estimation    
    local_kernel = EulerMaruyama(model, params, sde)
    local_sampler = Diffuser(local_kernel, grid, (sim.d_theta,), sde)
    
    # composition using GAUSS
    gauss_fn = GAUSSScoreFn(
        score_net=model, 
        params=params, 
        sde=sde, 
        prior=sim, 
        diffuser=local_sampler,
        marginal_prior_fn=lambda a, t: marginal_prior_score(a, t, sde)
    )

    # sample from final model
    kernel = EulerMaruyama(gauss_fn, params, sde)
    final_sampler = Diffuser(kernel, grid, (4,), sde)

    print(f"--- [INFER DEBUG] x_obs before vmap: {x_obs.shape} ---")

    return final_sampler.sample(key, x_obs, 1.0)

    # return jax.vmap(lambda k, x: final_sampler.sample(k, x, 1.0), in_axes=(0, None))(
    #     jax.random.split(key, 100), x_obs
    # )

# ALGORITHM 2: compose for noise injected time series
def infer_many(key, model, params, sim, x_obs):
    sde = SDE()
    grid = jnp.linspace(1e-3, 1.0, 100) 
    
    # 1. Create a local sampler for precision estimation    
    local_kernel = EulerMaruyama(model, params, sde)
    local_sampler = Diffuser(local_kernel, grid, (sim.d_theta,), sde)
    
    # composition using GAUSS
    gauss_fn = GAUSSScoreFn(
        score_net=model, 
        params=params, 
        sde=sde, 
        prior=sim, 
        diffuser=local_sampler,
        marginal_prior_fn=lambda a, t: marginal_prior_score(a, t, sde)
    )

    # sample from final model
    kernel = EulerMaruyama(gauss_fn, params, sde)
    final_sampler = Diffuser(kernel, grid, (4,), sde)

    print(f"--- [INFER DEBUG] x_obs before vmap: {x_obs.shape} ---")

    # 100개의 서로 다른 랜덤 키 생성
    keys = jax.random.split(key, 100)
    
    # jax.vmap을 사용해 100개의 샘플을 병렬로 생성
    # keys는 배치 축(0)으로 매핑하고, x_obs와 1.0(a_start)은 broadcast(None)
    samples = jax.vmap(final_sampler.sample, in_axes=(0, None, None))(keys, x_obs, 1.0)
    
    return samples

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # generate simulations
    lv = LotkaVolterra(
        dt=config['sim']['dt'], 
        sigma=config['sim']['sigma']
    ) 
    proposal = lv_prior_predictive_proposal(lv, key) 
    
    # LocalScoreNet: (B, T, d_x) -> (B, T-1, d_theta)
    model_inference = LocalScoreNet(config=config['model'])
    # # ScoreMLP: (B, 2*d_x) -> (B, d_theta)
    # model_engine = ScoreMLP(h_dim=config['model']['hidden_dim'], d_theta=config['model']['d_theta'])
    
    # Alg 1
    trained_params = train(key, model_inference, lv, proposal, config['train'])
    
    # Generate the Ground Truth data
    theta_true = jnp.array([[0.6, 0.025, 0.8, 0.025]]) # theta
    x0_true = jnp.array([[10.0, 5.0]])  # initial value

    from src.simulators import lv_traj
    x_obs_target = lv_traj(key, lv, theta_true, x0_true, T=20)
    print(f"--- [MAIN DEBUG] x_obs_target raw shape: {x_obs_target.shape} ---")

    # Alg 2
    samples = infer_many(key, model_inference, trained_params, lv, x_obs_target)  # x_obs_target (T, 2)
    print(f"completed sampling from the posterior: {samples.shape}")

    plot_posterior_samples(samples, theta_true)

