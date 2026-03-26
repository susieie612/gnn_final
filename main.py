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
from src.metrics import (Timer, posterior_accuracy_metrics, gauss_stability_metrics,
                         reverse_sde_diagnostics, posterior_predictive_metrics,
                         posterior_concentration_metrics, generate_metrics_table,
                         print_metrics_summary)
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



def save_results(result_dir, samples_raw, samples_pretransform, x_obs, theta_true,
                 train_losses, val_losses, num_samples, time_step, sim_name):
    """Save all inference results as .npy files for reuse without re-running."""
    os.makedirs(result_dir, exist_ok=True)
    np.save(os.path.join(result_dir, 'samples.npy'), np.array(samples_raw))
    np.save(os.path.join(result_dir, 'samples_pretransform.npy'), np.array(samples_pretransform))
    np.save(os.path.join(result_dir, 'x_obs.npy'), np.array(x_obs))
    np.save(os.path.join(result_dir, 'theta_true.npy'), np.array(theta_true))
    np.save(os.path.join(result_dir, 'train_losses.npy'), np.array(train_losses))
    if val_losses:
        val_steps, val_vals = zip(*val_losses)
        np.save(os.path.join(result_dir, 'val_steps.npy'), np.array(val_steps))
        np.save(os.path.join(result_dir, 'val_vals.npy'), np.array(val_vals))
    # save metadata
    np.savez(os.path.join(result_dir, 'meta.npz'),
             num_samples=num_samples, time_step=time_step, sim_name=sim_name)
    print(f"Results saved to {result_dir}/")


def load_results(result_dir):
    """Load previously saved results. Returns None if not found."""
    samples_path = os.path.join(result_dir, 'samples.npy')
    if not os.path.exists(samples_path):
        return None

    data = {
        'samples': jnp.array(np.load(os.path.join(result_dir, 'samples.npy'))),
        'samples_pretransform': np.load(os.path.join(result_dir, 'samples_pretransform.npy')),
        'x_obs': jnp.array(np.load(os.path.join(result_dir, 'x_obs.npy'))),
        'theta_true': jnp.array(np.load(os.path.join(result_dir, 'theta_true.npy'))),
        'train_losses': np.load(os.path.join(result_dir, 'train_losses.npy')).tolist(),
    }

    val_steps_path = os.path.join(result_dir, 'val_steps.npy')
    if os.path.exists(val_steps_path):
        val_steps = np.load(val_steps_path)
        val_vals = np.load(os.path.join(result_dir, 'val_vals.npy'))
        data['val_losses'] = list(zip(val_steps.tolist(), val_vals.tolist()))
    else:
        data['val_losses'] = []

    meta = np.load(os.path.join(result_dir, 'meta.npz'), allow_pickle=True)
    data['num_samples'] = int(meta['num_samples'])
    data['time_step'] = int(meta['time_step'])

    print(f"Loaded cached results from {result_dir}/")
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', action='store_true',
                        help='Force re-training and re-inference even if cached results exist')
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    sim_cfg = config['sim']
    timer = Timer()

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

    target_cfg = sim_cfg['target']
    time_step = target_cfg["time_steps"]

    # results/ directory: results/{SimName}/T{time_step}/
    result_dir = f"results/{active_sim_name}/T{time_step}"
    plot_dir = f"plots/{active_sim_name}"

    # ── Try loading cached results ──
    cached = None if args.rerun else load_results(result_dir)

    if cached is not None:
        # Use cached data — skip training & inference
        samples_raw = cached['samples']
        samples_raw_pretransform = cached['samples_pretransform']
        x_obs_target = cached['x_obs']
        theta_true = cached['theta_true']
        train_losses = cached['train_losses']
        val_losses = cached['val_losses']
        num_samples = cached['num_samples']
        final_sampler = None  # not available from cache

        param_transform = None
        if active_sim_name in ["LotkaVolterra", "SIR"]:
            param_transform = jnp.exp

    else:
        # ── Full pipeline: train + infer ──
        model_inference = LocalScoreNet(config=model_config)

        timer.start('training')
        trained_params, train_losses, val_losses = train(
            key, model_inference, sim, proposal, config['train'], sim_name=active_sim_name)
        timer.stop()

        theta_true = jnp.array([target_cfg['theta_true']])
        if target_cfg["x0_true"] is not None:
            x0_true = jnp.array([target_cfg["x0_true"]])
        else:
            _, x0_true = sim.x0(key, 1)

        from src.simulators import generate_trajectory
        x_obs_target = generate_trajectory(key, sim, theta_true, x0_true, T=time_step)

        print(f"--- Inference for {active_sim_name} with {time_step} time steps ---")
        timer.start('inference')
        samples_raw, final_sampler, num_samples = infer_many(
            key, model_inference, trained_params, sim, x_obs_target)
        timer.stop()

        samples_raw_pretransform = np.array(samples_raw)

        param_transform = None
        if active_sim_name in ["LotkaVolterra", "SIR"]:
            samples_raw = jnp.clip(samples_raw, a_min=-15.0, a_max=5.0)
            samples_raw = jnp.exp(samples_raw)
            param_transform = jnp.exp

        # ── Save results for future reuse ──
        save_results(result_dir, samples_raw, samples_raw_pretransform,
                     x_obs_target, theta_true, train_losses, val_losses,
                     num_samples, time_step, active_sim_name)

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

    # 6) Reverse SDE trajectory (only if sampler is available, i.e. not from cache)
    if final_sampler is not None:
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

    # ────────────────────────────────────────────────────────────
    # 8) Quantitative Evaluation Metrics
    # ────────────────────────────────────────────────────────────
    timer.start('metrics_computation')

    print("\nComputing quantitative metrics...")

    # 8a) Posterior accuracy vs ground truth
    accuracy = posterior_accuracy_metrics(samples_raw, theta_true)

    # 8b) Posterior concentration
    concentration = posterior_concentration_metrics(samples_raw, theta_true)

    # 8c) Reverse SDE divergence diagnostics (on pre-transform samples)
    divergence = reverse_sde_diagnostics(samples_raw_pretransform)

    # 8d) GAUSS composition stability (only when model is available)
    gauss_stab = {}
    if final_sampler is not None:
        sde = SDE()
        gauss_fn = GAUSSScoreFn(
            score_net=model_inference,
            params=trained_params,
            sde=sde,
            prior=sim,
            diffuser=final_sampler,
            marginal_prior_fn=lambda a, t: marginal_prior_score(a, t, sde, sim.prior_var),
            num_samples=min(200, max(50, 10 * sim.d_theta)),
        )
        probe_a_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        theta_probe = jnp.zeros(sim.d_theta)
        gauss_stab = gauss_stability_metrics(gauss_fn, probe_a_values, theta_probe, x_obs_target)
    else:
        print("  (GAUSS stability metrics skipped — using cached results)")

    # 8e) Posterior predictive metrics
    pred_metrics = posterior_predictive_metrics(
        key, sim, samples_raw, x_obs_target, param_transform=None, n_traj=50
    )

    timer.stop()

    # Collect timing
    timing = timer.summary()

    # Aggregate all metrics
    all_metrics = {
        'accuracy': accuracy,
        'concentration': concentration,
        'divergence': divergence,
        'gauss_stability': gauss_stab,
        'predictive': pred_metrics,
        'timing': timing,
        'swd_vs_prior': swd_val,
    }

    # Print to console
    print_metrics_summary(all_metrics, simulation_name=f"{active_sim_name} (T={time_step})")

    # Save metrics table as image
    generate_metrics_table(
        all_metrics,
        save_path=f"{plot_dir}/{active_sim_name}_{time_step}_N={num_samples}_metrics.png",
        simulation_name=f"{active_sim_name} (T={time_step})",
    )
