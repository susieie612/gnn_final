import jax
import jax.numpy as jnp
import optax
from src.simulators import lv_prior_predictive_proposal
from src.local_score_net import LocalScoreNet, ScoreMLP
from src.loss import denoising_score_matching_loss
from src.reverse import Diffuser, EulerMaruyama
from src.composition_score_fn import GAUSSScoreFn
from src.config import config
from src.visualize import plot_posterior_samples, visualise_local_transition, plot_reverse_trajectory
import src.simulators as sims

class SDE:
    def __init__(self, beta_min=0.1, beta_max=10.0):
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
        return jnp.sqrt(jnp.maximum(var, 1e-5)) # clip the variance

def marginal_prior_score(a, theta, sde):
    # # Denoising prior score p(theta_a)
    # return -theta
    from src.config import config
    active_sim = config['active_sim']
    var_theta = 1.0 if active_sim == "LotkaVolterra" else 3.0
    s_a = sde.mean_coeff(a)
    sigma_a = sde.std(a)
    
    # Denoising Prior Score: -theta / Var(theta_a)
    var_a = (s_a**2) * var_theta + (sigma_a**2)
    return -theta / (var_a + 1e-8)

# ALGORITHM 1: train local score
def train(key, model, sim, proposal_fn, config):
    sde = SDE()

    # get values from config dictionary
    lr = config.get('learning_rate', 1e-4)
    batch_size = config.get('batch_size', 128)
    optimizer = optax.adamw(learning_rate=lr) 

    d_x = sim.d_x
    d_theta = sim.d_theta
    
    # TODO: later update 2nd dim to match for general simulations
    init_x = jnp.ones((1, 2 * sim.d_x)) 
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
    for i in range(5000):
        key, k_theta, k_prop, k_sim = jax.random.split(key, 4)
        
        # theta sampling from prior distribution
        theta_raw = sim.prior(k_theta, batch_size)

        if active_sim_name in ["LotkaVolterra", "SIR"]:
            theta_sim = jnp.exp(theta_raw)
        else:
            theta_sim = theta_raw

        # sample x_t from proposal distribution p(x_t)
        _, x_t = proposal_fn(k_prop, batch_size)
        # do 1-step transition to get x_next
        x_next = sim.transition(k_sim, x_t, theta_sim) # (B, d_x)

        params, opt_state, loss = step(params, opt_state, key, theta_raw, x_t, x_next)
        
        if i % 1000 == 0: 
            print(f"Step {i}, Loss: {loss:.4f}")
    
    return params

# ALGORITHM 2: compose for noise injected time series
def infer(key, model, params, sim, x_obs):
    sde = SDE()
    # grid = jnp.linspace(1e-3, 1.0, 100) 
    grid = jnp.geompace(1e-2, 1.0, 100)  # for numerical stability near a=0
    
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
    final_sampler = Diffuser(kernel, grid, (sim.d_theta,), sde)

    # print(f"--- [INFER DEBUG] x_obs before vmap: {x_obs.shape} ---")

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
    final_sampler = Diffuser(kernel, grid, (sim.d_theta,), sde)

    num_samples = 30
    chunk_size = 10
    keys = jax.random.split(key, num_samples)

    sample_chunk_fn = jax.jit(jax.vmap(final_sampler.sample, in_axes=(0, None, None)))

    all_samples = []
    print("Sampling started...")
    for i in range(0, num_samples, chunk_size):
        chunk_keys = keys[i : i + chunk_size]
        
        # chunk sampling
        chunk_samples = sample_chunk_fn(chunk_keys, x_obs, 1.0)
        all_samples.append(chunk_samples)
        
        print(f"[{i + chunk_size}/{num_samples}] samples processed...")
        
    samples = jnp.concatenate(all_samples, axis=0)
    
    return samples, final_sampler


def get_simulator(key, sim_name, sim_params):
    sim_class = getattr(sims, sim_name)
    sim_instance = sim_class(**sim_params)

    if sim_name == "LotkaVolterra":
        proposal_fn = sims.lv_prior_predictive_proposal(sim_instance, key)
    elif sim_name == "KolmogorovFlow":
        proposal_fn = sims.kf_proposal(sim_instance)
    else: 
        proposal_fn = sim_instance.proposal

    return sim_instance, proposal_fn



if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    sim_cfg = config['sim']

    # load simulator
    active_sim_name = config['active_sim']
    sim_cfg = config['sim_settings'][active_sim_name]

    sim, proposal = get_simulator(
        key,
        sim_cfg['name'],
        sim_cfg['params']
    )

    model_config = config['model'].copy()
    model_config['d_theta'] = sim.d_theta
    model_config['d_x'] = sim.d_x

    model_inference = LocalScoreNet(
        config=model_config
        # hidden_dim = config['model']['hidden_dim'],
        # d_theta = sim.d_theta,
        # d_x = sim.d_x
    )

    # training (Alg 1)
    trained_params = train(key, model_inference, sim, proposal, config['train'])

    # visualize local transition training results
    # visualise_local_transition(key, model_inference, trained_params, sim, sde)


    # generate ground truth
    target_cfg = sim_cfg['target']
    theta_true = jnp.array([target_cfg['theta_true']])

    if target_cfg["x0_true"] is not None:
        x0_true = jnp.array([target_cfg["x0_true"]])
    else:
        _, x0_true = sim.x0(key, 1)

    # generate trajectory
    time_step = target_cfg["time_steps"] 

    from src.simulators import lv_traj
    x_obs_target = lv_traj(key, sim, theta_true, x0_true, T=time_step) # TODO: may need to treat KolmogorovFlow differently

    # inference (Alg 2)
    print(f"--- Inference for {active_sim_name} with {time_step} time steps ---")
    samples_raw, final_sampler = infer_many(key, model_inference, trained_params, sim, x_obs_target)
    if active_sim_name in ["LotkaVolterra", "SIR"]:
        samples_raw = jnp.clip(samples_raw, a_min=-15.0, a_max=5.0)
        samples_raw = jnp.exp(samples_raw)
    
    if active_sim_name in ["LotkaVolterra", "SIR"]:
        samples_raw = jnp.exp(samples_raw)

    # add visulaization tools here
    plot_posterior_samples(
        samples_raw, 
        theta_true, 
        save_path=f"plots/{active_sim_name}/{active_sim_name}_{time_step}_results.png",
        simulation_name=active_sim_name,
        simulation_length=time_step,
        )

    plot_reverse_trajectory(final_sampler, 
        key, 
        x_obs_target, 
        theta_true,
        save_path=f"plots/{active_sim_name}/{active_sim_name}_{time_step}_trajectory.png",
        )
