import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import math
import jax
import jax.numpy as jnp
import os



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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_loss(losses, save_path=None, simulation_name=None):
    """
    Plot training loss curve with log scale.
    Args:
        losses: list of loss values recorded during training
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses, color='steelblue', linewidth=0.8)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(losses, color='steelblue', linewidth=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Loss (log scale)')
    ax2.grid(True, alpha=0.3)

    if simulation_name:
        fig.suptitle(f'{simulation_name} Training', fontsize=13)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pairwise_posterior(samples, theta_true, param_names=None, save_path=None, simulation_name=None):
    """
    Corner plot: diagonal = 1D marginals, off-diagonal = 2D pairwise scatter.
    Args:
        samples: (num_samples, d_theta)
        theta_true: (1, d_theta) or (d_theta,)
    """
    d_theta = samples.shape[1]
    theta_true_flat = np.array(theta_true).flatten()

    if param_names is None:
        param_names = [f'theta_{i}' for i in range(d_theta)]

    fig, axes = plt.subplots(d_theta, d_theta, figsize=(3 * d_theta, 3 * d_theta))
    if d_theta == 1:
        axes = np.array([[axes]])

    for i in range(d_theta):
        for j in range(d_theta):
            ax = axes[i, j]

            if j > i:
                ax.axis('off')
                continue

            if i == j:
                ax.hist(samples[:, i], bins=30, alpha=0.6, color='steelblue',
                        edgecolor='white', density=True)
                ax.axvline(theta_true_flat[i], color='red', linestyle='--', linewidth=1.5)
                ax.set_ylabel('Density' if j == 0 else '')
            else:
                ax.scatter(samples[:, j], samples[:, i], alpha=0.3, s=8, color='steelblue')
                ax.axvline(theta_true_flat[j], color='red', linestyle='--', linewidth=1, alpha=0.7)
                ax.axhline(theta_true_flat[i], color='red', linestyle='--', linewidth=1, alpha=0.7)
                ax.plot(theta_true_flat[j], theta_true_flat[i], 'r+', markersize=12, markeredgewidth=2)

            if i == d_theta - 1:
                ax.set_xlabel(param_names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(param_names[i])
            elif j != 0:
                ax.set_yticklabels([])

    if simulation_name:
        fig.suptitle(f'{simulation_name} Pairwise Posterior', fontsize=14, y=1.01)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_posterior_predictive(key, sim, samples, x_obs, theta_true,
                              num_trajectories=20, param_transform=None,
                              save_path=None, simulation_name=None):
    """
    Posterior predictive check: simulate trajectories from posterior samples
    and compare with observed data.
    Args:
        sim: simulator instance
        samples: posterior samples (num_samples, d_theta)
        x_obs: observed trajectory (T+1, d_x)
        param_transform: optional fn to transform raw samples to sim params (e.g. jnp.exp)
    """
    from src.simulators import generate_trajectory

    T = x_obs.shape[0] - 1
    d_x = sim.d_x
    x0 = x_obs[0:1, :]

    n_samples = min(num_trajectories, samples.shape[0])
    idx = jax.random.choice(key, samples.shape[0], shape=(n_samples,), replace=False)
    selected = samples[np.array(idx)]

    trajectories = []
    for s in range(n_samples):
        k = jax.random.PRNGKey(s + 100)
        theta_s = selected[s:s+1, :]
        if param_transform is not None:
            theta_s = param_transform(theta_s)
        traj = generate_trajectory(k, sim, theta_s, x0, T)
        trajectories.append(np.array(traj))

    state_names = [f'x_{i}' for i in range(d_x)]
    cols = min(d_x, 3)
    rows = math.ceil(d_x / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
    time_axis = np.arange(T + 1)

    for d in range(d_x):
        ax = axes[d]
        for traj in trajectories:
            ax.plot(time_axis, traj[:, d], color='steelblue', alpha=0.15, linewidth=0.8)
        ax.plot(time_axis, np.array(x_obs[:, d]), color='red', linewidth=2, label='Observed')
        ax.set_xlabel('Time step')
        ax.set_ylabel(state_names[d])
        ax.set_title(state_names[d])
        if d == 0:
            ax.legend(['Posterior samples', 'Observed'])
        ax.grid(True, alpha=0.2)

    for j in range(d_x, len(axes)):
        fig.delaxes(axes[j])

    title = 'Posterior Predictive Check'
    if simulation_name:
        title = f'{simulation_name} {title}'
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_table(samples, theta_true, param_names=None, save_path=None, simulation_name=None):
    """
    Summary statistics table: mean, std, median, 90% CI, true value, |bias|.
    Args:
        samples: (num_samples, d_theta)
        theta_true: (1, d_theta) or (d_theta,)
    """
    d_theta = samples.shape[1]
    theta_true_flat = np.array(theta_true).flatten()

    if param_names is None:
        param_names = [f'theta_{i}' for i in range(d_theta)]

    fig, ax = plt.subplots(figsize=(max(8, 2 * d_theta), 1.5 + 0.5 * d_theta))
    ax.axis('off')

    col_labels = ['Param', 'True', 'Mean', 'Std', 'Median', '5%', '95%', '|Bias|']
    table_data = []
    for i in range(d_theta):
        s = np.array(samples[:, i])
        true_val = theta_true_flat[i]
        table_data.append([
            param_names[i],
            f'{true_val:.4f}',
            f'{np.mean(s):.4f}',
            f'{np.std(s):.4f}',
            f'{np.median(s):.4f}',
            f'{np.percentile(s, 5):.4f}',
            f'{np.percentile(s, 95):.4f}',
            f'{abs(np.mean(s) - true_val):.4f}',
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # highlight bias if larger than 1 std
    for i in range(d_theta):
        bias = abs(np.mean(np.array(samples[:, i])) - theta_true_flat[i])
        std = np.std(np.array(samples[:, i]))
        if std > 0 and bias / max(std, 1e-8) > 1.0:
            table[i + 1, 7].set_facecolor('#FFD6D6')

    title = 'Posterior Summary Statistics'
    if simulation_name:
        title = f'{simulation_name} {title}'
    ax.set_title(title, fontsize=13, pad=20)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


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
    plt.close()


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
    plt.close()