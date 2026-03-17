import matplotlib.pyplot as plt
import numpy as np
import math
import jax
import jax.numpy as jnp



def plot_posterior_samples(samples, theta_true, param_names=None, save_path=None, simulation_name=None, simulation_length=None):
    """
    Compare the inferred posterior and the ground truth
    """
    # samples: (num_samples, d_theta), theta_true: (1, d_theta)
    d_theta = samples.shape[1]
    theta_true_flat = np.array(theta_true).flatten()

    if param_names is None:
        param_names = [f'theta_{i}' for i in range(d_theta)]

    cols = min(d_theta, 4)
    rows = math.ceil(d_theta / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i in range(d_theta):
        # Posterior histogram
        axes[i].hist(samples[:, i], bins=30, alpha=0.6, color='skyblue', 
                     edgecolor='black', density=True, label='Inferred')
        # Ground Truth 
        axes[i].axvline(theta_true_flat[i], color='red', linestyle='--', 
                        linewidth=2, label='True Value')
        
        axes[i].set_title(f'Posterior of {param_names[i]}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if simulation_name is not None:
        if simulation_length is not None:
            plt.suptitle(f'{simulation_name} with T={simulation_length}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_reverse_trajectory(final_sampler, key, x_obs, theta_true, save_path=None):
    """
    visualize the a trajectory during the Reverse Diffusion
    """
    # to save Diffuser process
    grid = final_sampler.time_grid[::-1]
    std_a = final_sampler.sde.std(grid[0])
    k_init, _ = jax.random.split(key)
    
    # initial noise
    current_pos = std_a * jax.random.normal(k_init, final_sampler.theta_shape)
    state = final_sampler.kernel.init(current_pos, x_obs, grid[0])
    
    trajectory = [current_pos]
    times = [grid[0]]
    
    for a_new in grid[1:]:
        k_init, sk = jax.random.split(k_init)
        state = final_sampler.kernel.step(sk, state, a_new, x_obs)
        trajectory.append(state.position)
        times.append(a_new)
        
    trajectory = jnp.stack(trajectory) # (num_steps, d_theta)
    times = jnp.array(times)
    
    plt.figure(figsize=(10, 5))
    for i in range(trajectory.shape[1]):
        plt.plot(times, trajectory[:, i], label=f'theta_{i}')
        if theta_true is not None:
            plt.axhline(y=theta_true[0, i], color='r', linestyle='--', alpha=0.5)
            
    plt.xlim(1.0, 0.0) # reversed to match the direction of a
    plt.xlabel('Diffusion Time (a)')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Trajectory during Reverse SDE')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualise_local_transition(key, model, params, sim, sde, n_samples=500):
    """
    for chekcing the training on single transition (x_t -> x_{t+1})
    local posterior sampling + visualization
    """
    # ground truth
    k1, k2, k3, k4 = jax.random.split(key, 4)
    theta_true = sim.prior(k1, 1) # (1, d_theta)
    
    # initialize x_t sampled from proposal
    _, x_t = sim.proposal(k2, 1) # (1, d_x)
    x_next = sim.transition(k3, x_t, theta_true) # (1, d_x)
    
    # local sampler
    # test ScoreMLP
    from src.reverse import Diffuser, EulerMaruyama
    grid = jnp.linspace(1e-3, 1.0, 100)
    kernel = EulerMaruyama(model, params, sde)
    local_sampler = Diffuser(kernel, grid, (sim.d_theta,), sde)
    
    # input x_pair for ScoreMLP 
    x_pair = jnp.concatenate([x_t, x_next], axis=-1).squeeze(0) # (2 * d_x)
    
    # local posterior sampling
    print(f"Sampling local posterior for theta_true: {theta_true}...")
    sample_fn = jax.jit(jax.vmap(lambda k: local_sampler.sample(k, x_pair, 1.0)))
    keys = jax.random.split(k4, n_samples)
    samples = sample_fn(keys)
    
    # visualize
    d_theta = sim.d_theta
    fig, axes = plt.subplots(1, d_theta, figsize=(4 * d_theta, 4))
    if d_theta == 1: axes = [axes]
    
    for i in range(d_theta):
        axes[i].hist(samples[:, i], bins=30, alpha=0.5, color='skyblue', density=True, label='Local Samples')
        axes[i].axvline(theta_true[0, i], color='red', linestyle='--', label='True Value')
        axes[i].set_title(f'Local Posterior: Param {i}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()