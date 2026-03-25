import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
from src.simulators import lv_prior_predictive_proposal
from src.local_score_net import LocalScoreNet, ScoreMLP
from src.loss import denoising_score_matching_loss
from src.reverse import Diffuser, EulerMaruyama
from src.composition_score_fn import GAUSSScoreFn
from src.config import config
from src.visualize import (plot_posterior_samples, visualise_local_transition,
                           plot_reverse_trajectory, plot_training_loss,
                           plot_pairwise_posterior, plot_posterior_predictive,
                           plot_summary_table, compute_swd, compute_swd_vs_prior)
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

def marginal_prior_score(a, theta, sde, var_theta=1.0):
    """Marginal prior score: ∇log p_a(θ_a) for N(0, var_theta) prior."""
    s_a = sde.mean_coeff(a)
    sigma_a = sde.std(a)

    # p_a(θ_a) = N(0, s_a² * var_theta + σ_a²)
    var_a = (s_a**2) * var_theta + (sigma_a**2)
    return -theta / (var_a + 1e-8)

# ALGORITHM 1: train local score
def train(key, model, sim, proposal_fn, config, sim_name=None):
    sde = SDE()

    # get values from config dictionary
    lr = config.get('learning_rate', 1e-4)
    num_steps = config.get('num_steps', 5000)
    batch_size = config.get('batch_size', 128)
    optimizer = optax.adamw(learning_rate=lr)

    d_x = sim.d_x
    d_theta = sim.d_theta

    init_x = jnp.ones((1, 2 * sim.d_x))
    init_theta = jnp.ones((1, sim.d_theta))
    init_a = jnp.ones((1,))

    # create the network
    params = model.init(key, init_x, init_theta, init_a)['params']
    opt_state = optimizer.init(params)

    # simulators with log-space parameters
    log_param_sims = ["LotkaVolterra", "SIR"]

    @jax.jit
    def step(params, opt_state, k, theta, x_t, x_next):
        x_train = jnp.stack([x_t, x_next], axis=1) # (B, 2, d_x)

        loss, grads = jax.value_and_grad(denoising_score_matching_loss)(
            params, model, k, theta, x_train, sde
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_loss(params, k, theta, x_t, x_next):
        x_train = jnp.stack([x_t, x_next], axis=1)
        return denoising_score_matching_loss(params, model, k, theta, x_train, sde)

    # generate fixed validation set (once)
    val_key = jax.random.PRNGKey(999)
    vk1, vk2, vk3, vk4 = jax.random.split(val_key, 4)
    val_theta_raw = sim.prior(vk1, batch_size)
    val_theta_sim = jnp.exp(val_theta_raw) if sim_name in log_param_sims else val_theta_raw
    _, val_x_t = proposal_fn(vk2, batch_size)
    val_x_next = sim.transition(vk3, val_x_t, val_theta_sim)

    val_interval = config.get('val_interval', 500)
    patience = config.get('patience', 2000)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_params = None
    steps_without_improve = 0

    print("Starting algorithm 1")
    for i in range(num_steps):
        key, k_theta, k_prop, k_sim = jax.random.split(key, 4)

        # theta sampling from prior distribution
        theta_raw = sim.prior(k_theta, batch_size)

        if sim_name in log_param_sims:
            theta_sim = jnp.exp(theta_raw)
        else:
            theta_sim = theta_raw

        # sample x_t from proposal distribution p(x_t)
        _, x_t = proposal_fn(k_prop, batch_size)
        # do 1-step transition to get x_next
        x_next = sim.transition(k_sim, x_t, theta_sim) # (B, d_x)

        params, opt_state, loss = step(params, opt_state, key, theta_raw, x_t, x_next)
        train_losses.append(float(loss))

        # validation
        if i % val_interval == 0:
            v_loss = float(eval_loss(params, vk4, val_theta_raw, val_x_t, val_x_next))
            val_losses.append((i, v_loss))

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_params = jax.tree.map(lambda x: x.copy(), params)
                steps_without_improve = 0
            else:
                steps_without_improve += val_interval

            if i % 1000 == 0:
                print(f"Step {i}, Train: {loss:.4f}, Val: {v_loss:.4f}"
                      f"{' *best*' if steps_without_improve == 0 else ''}")

            # early stopping
            if steps_without_improve >= patience and i > patience:
                print(f"Early stopping at step {i} (no improvement for {patience} steps)")
                break
        elif i % 1000 == 0:
            print(f"Step {i}, Train: {loss:.4f}")

    # return best params if validation was used
    final_params = best_params if best_params is not None else params
    return final_params, train_losses, val_losses

# ALGORITHM 2: compose for noise injected time series
def infer(key, model, params, sim, x_obs):
    sde = SDE()
    grid = jnp.geomspace(1e-2, 1.0, 100)  # for numerical stability near a=0

    # 1. Create a local sampler for precision estimation
    local_kernel = EulerMaruyama(model, params, sde)
    local_sampler = Diffuser(local_kernel, grid, (sim.d_theta,), sde)

    prec_samples = min(200, max(50, 10 * sim.d_theta))

    # composition using GAUSS
    gauss_fn = GAUSSScoreFn(
        score_net=model,
        params=params,
        sde=sde,
        prior=sim,
        diffuser=local_sampler,
        marginal_prior_fn=lambda a, t: marginal_prior_score(a, t, sde, sim.prior_var),
        num_samples=prec_samples,
    )

    # sample from final model
    kernel = EulerMaruyama(gauss_fn, params, sde)
    final_sampler = Diffuser(kernel, grid, (sim.d_theta,), sde)

    return final_sampler.sample(key, x_obs, 1.0)

# ALGORITHM 2: compose for noise injected time series
def infer_many(key, model, params, sim, x_obs):
    sde = SDE()
    grid = jnp.geomspace(1e-2, 1.0, 100)  # for numerical stability near a=0

    # 1. Create a local sampler for precision estimation
    local_kernel = EulerMaruyama(model, params, sde)
    local_sampler = Diffuser(local_kernel, grid, (sim.d_theta,), sde)

    # precision estimation: fewer samples for speed, more for accuracy
    prec_samples = min(200, max(50, 10 * sim.d_theta))

    # composition using GAUSS
    gauss_fn = GAUSSScoreFn(
        score_net=model,
        params=params,
        sde=sde,
        prior=sim,
        diffuser=local_sampler,
        marginal_prior_fn=lambda a, t: marginal_prior_score(a, t, sde, sim.prior_var),
        num_samples=prec_samples,
    )

    # sample from final model
    kernel = EulerMaruyama(gauss_fn, params, sde)
    final_sampler = Diffuser(kernel, grid, (sim.d_theta,), sde)

    num_samples = 100
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
    
    return samples, final_sampler, num_samples


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
    trained_params, train_losses, val_losses = train(key, model_inference, sim, proposal, config['train'], sim_name=active_sim_name)

    plot_dir = f"plots/{active_sim_name}"

    # generate ground truth
    target_cfg = sim_cfg['target']
    theta_true = jnp.array([target_cfg['theta_true']])

    if target_cfg["x0_true"] is not None:
        x0_true = jnp.array([target_cfg["x0_true"]])
    else:
        _, x0_true = sim.x0(key, 1)

    # generate trajectory
    time_step = target_cfg["time_steps"]

    from src.simulators import generate_trajectory
    x_obs_target = generate_trajectory(key, sim, theta_true, x0_true, T=time_step)

    # inference (Alg 2)
    print(f"--- Inference for {active_sim_name} with {time_step} time steps ---")
    samples_raw, final_sampler, num_samples = infer_many(key, model_inference, trained_params, sim, x_obs_target)

    # transform for log-space simulators
    param_transform = None
    if active_sim_name in ["LotkaVolterra", "SIR"]:
        samples_raw = jnp.clip(samples_raw, a_min=-15.0, a_max=5.0)
        samples_raw = jnp.exp(samples_raw)
        param_transform = jnp.exp

    os.makedirs(plot_dir, exist_ok=True)

    # 1) Training loss curve
    plot_training_loss(
        train_losses,
        val_losses=val_losses,
        save_path=f"{plot_dir}/{active_sim_name}_{time_step}_N={num_samples}_training_loss.png",
        simulation_name=active_sim_name,
        simulation_length=time_step,
    )

    # 2) 1D marginal posterior histograms
    plot_posterior_samples(
        samples_raw,
        theta_true,
        save_path=f"{plot_dir}/{active_sim_name}_{time_step}_N={num_samples}_posterior.png",
        simulation_name=active_sim_name,
        simulation_length=time_step,
    )

    # 3) Pairwise posterior (corner plot)
    if sim.d_theta >= 2:
        plot_pairwise_posterior(
            samples_raw,
            theta_true,
            save_path=f"{plot_dir}/{active_sim_name}_{time_step}_N={num_samples}_corner.png",
            simulation_name=active_sim_name,
        )

    # 4) Summary statistics table
    plot_summary_table(
        samples_raw,
        theta_true,
        save_path=f"{plot_dir}/{active_sim_name}_{time_step}_N={num_samples}_summary.png",
        simulation_name=active_sim_name,
    )

    # 5) Posterior predictive check
    plot_posterior_predictive(
        key, sim, samples_raw, x_obs_target, theta_true,
        num_trajectories=20,
        param_transform=None,
        save_path=f"{plot_dir}/{active_sim_name}_{time_step}_N={num_samples}_predictive.png",
        simulation_name=active_sim_name,
    )

    # 6) Reverse SDE trajectory
    plot_reverse_trajectory(
        final_sampler,
        key,
        x_obs_target,
        theta_true,
        save_path=f"{plot_dir}/{active_sim_name}_{time_step}_N={num_samples}_trajectory.png",
    )

    # 7) Sliced Wasserstein Distance (posterior vs prior)
    swd_val, _ = compute_swd_vs_prior(key, sim, samples_raw)
    print(f"\n{'='*50}")
    print(f"SWD (Posterior vs Prior): {swd_val:.6f}")
    print(f"  - Higher = posterior is more concentrated (good)")
    print(f"  - Near 0 = posterior ≈ prior (uninformative)")
    print(f"{'='*50}")
