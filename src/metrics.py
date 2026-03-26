"""
Quantitative evaluation metrics for simulation-based inference.
Replaces ABC-dependent metrics (C2ST, SW-1) with self-contained diagnostics.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os


# ──────────────────────────────────────────────────────────────
# 1. Posterior Accuracy Metrics (vs ground truth)
# ──────────────────────────────────────────────────────────────

def posterior_accuracy_metrics(samples, theta_true):
    """
    Compute accuracy metrics between posterior samples and ground truth.

    Returns dict with:
      - l2_norm: ||E[θ] - θ*||_2
      - per_param_bias: per-parameter signed bias
      - per_param_std: posterior standard deviation
      - rmse: root mean squared error per param
      - relative_bias: |bias| / |θ*| (where θ* != 0)
      - z_score: |bias| / std  (how many stds away from truth)
      - coverage_90: fraction of params where θ* falls in 90% CI
      - p_values: two-sided p-value per param (empirical quantile test)
      - ess: effective sample size per param (using autocorrelation)
    """
    samples = np.array(samples)
    theta_true_flat = np.array(theta_true).flatten()
    d_theta = samples.shape[1]

    means = np.mean(samples, axis=0)
    stds = np.std(samples, axis=0)
    bias = means - theta_true_flat

    # L2 norm of bias vector
    l2_norm = np.linalg.norm(bias)

    # RMSE per parameter
    rmse = np.sqrt(np.mean((samples - theta_true_flat[np.newaxis, :]) ** 2, axis=0))

    # Relative bias (avoid division by zero)
    rel_bias = np.where(
        np.abs(theta_true_flat) > 1e-8,
        np.abs(bias) / np.abs(theta_true_flat),
        np.abs(bias)
    )

    # Z-score
    z_score = np.abs(bias) / np.maximum(stds, 1e-8)

    # 90% credible interval coverage
    q5 = np.percentile(samples, 5, axis=0)
    q95 = np.percentile(samples, 95, axis=0)
    coverage_90 = np.mean((theta_true_flat >= q5) & (theta_true_flat <= q95))

    # Empirical p-value: fraction of samples more extreme than truth
    p_values = np.zeros(d_theta)
    for i in range(d_theta):
        rank = np.mean(samples[:, i] <= theta_true_flat[i])
        p_values[i] = 2 * min(rank, 1 - rank)  # two-sided

    # Effective sample size (batch means ESS)
    ess = _compute_ess(samples)

    return {
        'l2_norm': float(l2_norm),
        'per_param_bias': bias.tolist(),
        'per_param_std': stds.tolist(),
        'rmse': rmse.tolist(),
        'relative_bias': rel_bias.tolist(),
        'z_score': z_score.tolist(),
        'coverage_90': float(coverage_90),
        'p_values': p_values.tolist(),
        'ess': ess.tolist(),
    }


def _compute_ess(samples):
    """Effective sample size using initial monotone sequence estimator."""
    n, d = samples.shape
    ess = np.zeros(d)
    for i in range(d):
        x = samples[:, i]
        x = x - np.mean(x)
        # autocorrelation via FFT
        f = np.fft.fft(x, n=2*n)
        acf = np.fft.ifft(f * np.conj(f))[:n].real
        acf = acf / acf[0]
        # sum pairs until negative
        tau = 1.0
        for k in range(1, n // 2):
            pair_sum = acf[2*k - 1] + acf[2*k]
            if pair_sum < 0:
                break
            tau += 2 * pair_sum
        ess[i] = n / max(tau, 1.0)
    return ess


# ──────────────────────────────────────────────────────────────
# 2. GAUSS Composition Stability Metrics
# ──────────────────────────────────────────────────────────────

def gauss_stability_metrics(gauss_fn, a_values, theta_a, x_obs):
    """
    Evaluate GAUSS composition stability at multiple diffusion times.

    Returns dict with:
      - eigenvalue_correction_ratio: fraction of diffusion times where
        eigenvalue correction (PD enforcement) was needed
      - mean_condition_number: average condition number of lambda_a
      - per_a_details: list of per-step diagnostics
    """
    x_0_T = x_obs
    if x_0_T.ndim != 3:
        x_0_T = x_0_T[jnp.newaxis, ...]

    x_pairs = jnp.concatenate([x_0_T[:, :-1, :], x_0_T[:, 1:, :]], axis=-1)
    x_pairs = jnp.squeeze(x_pairs, axis=0)
    num_transitions = x_pairs.shape[0]

    corrections_needed = 0
    condition_numbers = []
    neg_eig_fractions = []
    details = []

    for a in a_values:
        a_jnp = jnp.array(a)

        # Get precisions
        sigma_a_inv = gauss_fn.get_prior_precision(a_jnp)
        sigma_a_t_inv = gauss_fn.estimate_local_precision(a_jnp, theta_a, x_0_T)

        # Raw lambda (before PD correction)
        lambda_raw = jnp.sum(sigma_a_t_inv, axis=0) + (1 - num_transitions) * sigma_a_inv
        lambda_raw = (lambda_raw + lambda_raw.T) / 2.0
        eigenvalues = np.array(jnp.linalg.eigvalsh(lambda_raw))

        n_negative = int(np.sum(eigenvalues < 0))
        total_eigs = len(eigenvalues)
        needs_correction = n_negative > 0

        if needs_correction:
            corrections_needed += 1

        # Condition number of corrected lambda
        score = gauss_fn(a_jnp, theta_a, x_obs, ensure_pd=True)
        lambda_corrected = jnp.sum(sigma_a_t_inv, axis=0) + (1 - num_transitions) * sigma_a_inv
        eigs_corrected = np.array(jnp.linalg.eigvalsh((lambda_corrected + lambda_corrected.T) / 2))
        cond = float(np.max(np.abs(eigs_corrected)) / max(np.min(np.abs(eigs_corrected)), 1e-10))
        condition_numbers.append(cond)

        neg_eig_fractions.append(n_negative / total_eigs)

        details.append({
            'a': float(a),
            'needs_correction': needs_correction,
            'n_negative_eigenvalues': n_negative,
            'neg_eig_fraction': n_negative / total_eigs,
            'condition_number': cond,
            'eigenvalue_range': [float(eigenvalues.min()), float(eigenvalues.max())],
        })

    return {
        'eigenvalue_correction_ratio': corrections_needed / len(a_values),
        'mean_condition_number': float(np.mean(condition_numbers)),
        'median_condition_number': float(np.median(condition_numbers)),
        'mean_neg_eig_fraction': float(np.mean(neg_eig_fractions)),
        'per_a_details': details,
    }


# ──────────────────────────────────────────────────────────────
# 3. Reverse SDE Divergence Diagnostics
# ──────────────────────────────────────────────────────────────

def reverse_sde_diagnostics(samples_raw):
    """
    Diagnose reverse SDE sampling quality.

    Args:
        samples_raw: raw posterior samples BEFORE any transform (N, d_theta)

    Returns dict with:
      - divergence_ratio_1e2: fraction with ||θ|| > 100
      - divergence_ratio_1e1: fraction with ||θ|| > 10
      - max_norm: maximum L2 norm across samples
      - mean_norm: average L2 norm
      - median_norm: median L2 norm
      - per_param_range: [min, max] per parameter
      - nan_ratio: fraction of samples containing NaN
      - inf_ratio: fraction of samples containing Inf
    """
    samples = np.array(samples_raw)
    n = samples.shape[0]

    norms = np.linalg.norm(samples, axis=1)

    nan_mask = np.any(np.isnan(samples), axis=1)
    inf_mask = np.any(np.isinf(samples), axis=1)
    valid_mask = ~nan_mask & ~inf_mask

    valid_norms = norms[valid_mask] if np.any(valid_mask) else norms

    per_param_range = []
    for i in range(samples.shape[1]):
        col = samples[valid_mask, i] if np.any(valid_mask) else samples[:, i]
        per_param_range.append([float(np.min(col)), float(np.max(col))])

    return {
        'divergence_ratio_1e2': float(np.mean(norms > 100)),
        'divergence_ratio_1e1': float(np.mean(norms > 10)),
        'max_norm': float(np.max(norms)) if len(norms) > 0 else float('nan'),
        'mean_norm': float(np.mean(valid_norms)),
        'median_norm': float(np.median(valid_norms)),
        'per_param_range': per_param_range,
        'nan_ratio': float(np.mean(nan_mask)),
        'inf_ratio': float(np.mean(inf_mask)),
        'num_valid_samples': int(np.sum(valid_mask)),
    }


# ──────────────────────────────────────────────────────────────
# 4. Timing / Efficiency Metrics
# ──────────────────────────────────────────────────────────────

class Timer:
    """Context manager to record wall-clock time for stages."""
    def __init__(self):
        self.records = {}
        self._start = None
        self._label = None

    def start(self, label):
        self._label = label
        self._start = time.time()

    def stop(self):
        if self._start is not None and self._label is not None:
            elapsed = time.time() - self._start
            self.records[self._label] = elapsed
            self._start = None
            self._label = None

    def summary(self):
        """Return timing summary dict."""
        total = sum(self.records.values())
        result = {}
        for k, v in self.records.items():
            result[k] = {
                'seconds': round(v, 2),
                'fraction': round(v / total, 3) if total > 0 else 0,
            }
        result['total'] = {'seconds': round(total, 2), 'fraction': 1.0}
        return result


# ──────────────────────────────────────────────────────────────
# 5. Additional Diagnostics
# ──────────────────────────────────────────────────────────────

def posterior_predictive_metrics(key, sim, samples, x_obs, param_transform=None, n_traj=50):
    """
    Quantitative posterior predictive check.

    Returns:
      - mean_trajectory_rmse: avg RMSE between predicted and observed trajectories
      - trajectory_coverage: fraction of observed points within 90% predictive interval
      - energy_distance: energy distance between observed and predicted trajectories
    """
    from src.simulators import generate_trajectory

    T = x_obs.shape[0] - 1
    x0 = x_obs[0:1, :]

    n_samples = min(n_traj, samples.shape[0])
    idx = jax.random.choice(key, samples.shape[0], shape=(n_samples,), replace=False)
    selected = samples[np.array(idx)]

    trajectories = []
    for s in range(n_samples):
        k = jax.random.PRNGKey(s + 200)
        theta_s = selected[s:s+1, :]
        if param_transform is not None:
            theta_s = param_transform(theta_s)
        traj = generate_trajectory(k, sim, theta_s, x0, T)
        trajectories.append(np.array(traj))

    trajectories = np.stack(trajectories)  # (n_samples, T+1, d_x)
    x_obs_np = np.array(x_obs)

    # RMSE per trajectory, then average
    rmses = np.sqrt(np.mean((trajectories - x_obs_np[np.newaxis, :, :]) ** 2, axis=(1, 2)))

    # Predictive coverage: fraction of observed points within [5%, 95%] predictive interval
    q5 = np.percentile(trajectories, 5, axis=0)
    q95 = np.percentile(trajectories, 95, axis=0)
    within = (x_obs_np >= q5) & (x_obs_np <= q95)
    coverage = float(np.mean(within))

    # Calibration score (mean absolute deviation of coverage from nominal across dimensions)
    pred_mean = np.mean(trajectories, axis=0)
    pred_std = np.std(trajectories, axis=0) + 1e-8
    standardized = (x_obs_np - pred_mean) / pred_std
    calibration_score = float(np.mean(np.abs(standardized)))

    return {
        'mean_trajectory_rmse': float(np.mean(rmses)),
        'std_trajectory_rmse': float(np.std(rmses)),
        'predictive_coverage_90': coverage,
        'calibration_score': calibration_score,
    }


def posterior_concentration_metrics(samples, theta_true):
    """
    Measures how well the posterior concentrates around ground truth.

    Returns:
      - mahalanobis_distance: Mahalanobis distance of θ* from posterior
      - kl_divergence_gaussian: KL(posterior_gaussian || prior_gaussian)
        treating posterior as Gaussian fit
      - posterior_entropy: differential entropy of Gaussian-fitted posterior
    """
    samples = np.array(samples)
    theta_true_flat = np.array(theta_true).flatten()
    d = samples.shape[1]

    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov_reg = cov + 1e-6 * np.eye(d)

    # Mahalanobis distance
    diff = theta_true_flat - mean
    cov_inv = np.linalg.inv(cov_reg)
    mahala = float(np.sqrt(diff @ cov_inv @ diff))

    # KL from Gaussian posterior to unit Gaussian prior:
    # KL(N(μ,Σ) || N(0,I)) = 0.5 * (tr(Σ) + μᵀμ - d - ln|Σ|)
    sign, logdet = np.linalg.slogdet(cov_reg)
    kl = 0.5 * (np.trace(cov_reg) + mean @ mean - d - logdet)

    # Differential entropy of Gaussian
    entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * logdet

    # Posterior correlation matrix
    stds = np.sqrt(np.diag(cov_reg))
    corr = cov_reg / (stds[:, None] * stds[None, :] + 1e-10)
    max_offdiag_corr = 0.0
    if d > 1:
        mask = ~np.eye(d, dtype=bool)
        max_offdiag_corr = float(np.max(np.abs(corr[mask])))

    return {
        'mahalanobis_distance': mahala,
        'kl_div_to_prior': float(kl),
        'posterior_entropy': float(entropy),
        'max_offdiag_correlation': max_offdiag_corr,
    }


# ──────────────────────────────────────────────────────────────
# 6. Table Generation (save as image + print to console)
# ──────────────────────────────────────────────────────────────

def generate_metrics_table(all_metrics, param_names=None, save_path=None, simulation_name=None):
    """
    Generate a comprehensive metrics table as a matplotlib figure.

    Args:
        all_metrics: dict with keys from the metric functions above
        param_names: list of parameter names
        save_path: path to save the figure
        simulation_name: name for title
    """
    accuracy = all_metrics.get('accuracy', {})
    divergence = all_metrics.get('divergence', {})
    gauss = all_metrics.get('gauss_stability', {})
    timing = all_metrics.get('timing', {})
    concentration = all_metrics.get('concentration', {})
    predictive = all_metrics.get('predictive', {})

    d_theta = len(accuracy.get('per_param_bias', []))
    if param_names is None:
        param_names = [f'θ_{i}' for i in range(d_theta)]

    # ── Section 1: Per-parameter accuracy ──
    section1_headers = ['Param', 'Bias', 'Std', 'RMSE', 'Rel. Bias', 'z-score', 'p-value', 'ESS']
    section1_data = []
    for i in range(d_theta):
        section1_data.append([
            param_names[i],
            f"{accuracy['per_param_bias'][i]:+.4f}",
            f"{accuracy['per_param_std'][i]:.4f}",
            f"{accuracy['rmse'][i]:.4f}",
            f"{accuracy['relative_bias'][i]:.3f}",
            f"{accuracy['z_score'][i]:.2f}",
            f"{accuracy['p_values'][i]:.3f}",
            f"{accuracy['ess'][i]:.1f}",
        ])

    # ── Section 2: Global summary ──
    section2_headers = ['Metric', 'Value']
    section2_data = [
        ['L2 Norm (bias)', f"{accuracy.get('l2_norm', 0):.4f}"],
        ['90% CI Coverage', f"{accuracy.get('coverage_90', 0):.2%}"],
        ['Mahalanobis Distance', f"{concentration.get('mahalanobis_distance', 0):.3f}"],
        ['KL(posterior || prior)', f"{concentration.get('kl_div_to_prior', 0):.3f}"],
        ['Posterior Entropy', f"{concentration.get('posterior_entropy', 0):.3f}"],
        ['Max Off-diag |ρ|', f"{concentration.get('max_offdiag_correlation', 0):.3f}"],
    ]

    # ── Section 3: Reverse SDE diagnostics ──
    section3_headers = ['Metric', 'Value']
    section3_data = [
        ['Diverged (‖θ‖>100)', f"{divergence.get('divergence_ratio_1e2', 0):.2%}"],
        ['Diverged (‖θ‖>10)', f"{divergence.get('divergence_ratio_1e1', 0):.2%}"],
        ['Max ‖θ‖', f"{divergence.get('max_norm', 0):.2f}"],
        ['Mean ‖θ‖', f"{divergence.get('mean_norm', 0):.2f}"],
        ['NaN ratio', f"{divergence.get('nan_ratio', 0):.2%}"],
        ['Valid samples', f"{divergence.get('num_valid_samples', 0)}"],
    ]

    # ── Section 4: GAUSS stability ──
    section4_headers = ['Metric', 'Value']
    section4_data = [
        ['Eigenvalue Correction Ratio', f"{gauss.get('eigenvalue_correction_ratio', 0):.2%}"],
        ['Mean Condition Number', f"{gauss.get('mean_condition_number', 0):.1f}"],
        ['Median Condition Number', f"{gauss.get('median_condition_number', 0):.1f}"],
        ['Mean Neg. Eigenvalue Frac.', f"{gauss.get('mean_neg_eig_fraction', 0):.2%}"],
    ]

    # ── Section 5: Timing ──
    section5_headers = ['Stage', 'Time (s)', 'Fraction']
    section5_data = []
    for stage, info in timing.items():
        section5_data.append([
            stage,
            f"{info['seconds']:.1f}",
            f"{info['fraction']:.1%}",
        ])

    # ── Section 6: Predictive ──
    section6_headers = ['Metric', 'Value']
    section6_data = [
        ['Mean Traj. RMSE', f"{predictive.get('mean_trajectory_rmse', 0):.4f}"],
        ['Std Traj. RMSE', f"{predictive.get('std_trajectory_rmse', 0):.4f}"],
        ['Pred. Coverage (90%)', f"{predictive.get('predictive_coverage_90', 0):.2%}"],
        ['Calibration Score', f"{predictive.get('calibration_score', 0):.3f}"],
    ]

    # ── Render ──
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    title = 'Quantitative Evaluation Metrics'
    if simulation_name:
        title = f'{simulation_name} — {title}'
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)

    def _draw_table(ax, headers, data, section_title, highlight_col=None):
        ax.axis('off')
        ax.set_title(section_title, fontsize=11, fontweight='bold', pad=10, loc='left')
        if not data:
            ax.text(0.5, 0.5, '(no data)', ha='center', va='center', fontsize=10, color='gray')
            return
        tbl = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.4)
        for j in range(len(headers)):
            tbl[0, j].set_facecolor('#2c3e50')
            tbl[0, j].set_text_props(color='white', fontweight='bold')
        # Highlight z-score > 2 or p-value < 0.05
        if highlight_col is not None:
            for i in range(len(data)):
                try:
                    val = float(data[i][highlight_col])
                    if highlight_col == 5 and val > 2.0:  # z-score
                        tbl[i + 1, highlight_col].set_facecolor('#FFD6D6')
                    if highlight_col == 6 and val < 0.05:  # p-value
                        tbl[i + 1, highlight_col].set_facecolor('#FFD6D6')
                except (ValueError, IndexError):
                    pass

    ax1 = fig.add_subplot(gs[0, :])
    _draw_table(ax1, section1_headers, section1_data, '① Per-Parameter Posterior Accuracy', highlight_col=5)

    ax2 = fig.add_subplot(gs[1, 0])
    _draw_table(ax2, section2_headers, section2_data, '② Global Posterior Quality')

    ax3 = fig.add_subplot(gs[1, 1])
    _draw_table(ax3, section3_headers, section3_data, '③ Reverse SDE Diagnostics')

    ax4 = fig.add_subplot(gs[2, 0])
    _draw_table(ax4, section4_headers, section4_data, '④ GAUSS Composition Stability')

    ax5 = fig.add_subplot(gs[2, 1])
    # Combine timing + predictive
    combined_headers = ['Metric', 'Value']
    combined_data = section5_data[:]
    for row in section5_data:
        combined_data_entry = [row[0], f"{row[1]}s ({row[2]})"]
    combined_data = [[r[0], f"{r[1]}s ({r[2]})"] for r in section5_data]
    combined_data.append(['─── Predictive ───', ''])
    combined_data.extend(section6_data)
    _draw_table(ax5, combined_headers, combined_data, '⑤ Efficiency & Predictive Check')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return all_metrics


def print_metrics_summary(all_metrics, simulation_name=None):
    """Print a concise text summary to console."""
    header = f"{'='*60}"
    title = f"  METRICS: {simulation_name}" if simulation_name else "  METRICS SUMMARY"

    print(f"\n{header}")
    print(title)
    print(header)

    acc = all_metrics.get('accuracy', {})
    print(f"\n[Posterior Accuracy]")
    print(f"  L2 Norm (bias):      {acc.get('l2_norm', 0):.4f}")
    print(f"  90% CI Coverage:     {acc.get('coverage_90', 0):.2%}")
    d = len(acc.get('per_param_bias', []))
    for i in range(d):
        p_str = f"  θ_{i}: bias={acc['per_param_bias'][i]:+.4f}, "
        p_str += f"std={acc['per_param_std'][i]:.4f}, "
        p_str += f"z={acc['z_score'][i]:.2f}, "
        p_str += f"p={acc['p_values'][i]:.3f}"
        flag = " ⚠" if acc['z_score'][i] > 2 else ""
        print(p_str + flag)

    conc = all_metrics.get('concentration', {})
    print(f"\n[Posterior Concentration]")
    print(f"  Mahalanobis Dist:    {conc.get('mahalanobis_distance', 0):.3f}")
    print(f"  KL(post || prior):   {conc.get('kl_div_to_prior', 0):.3f}")

    div = all_metrics.get('divergence', {})
    print(f"\n[Reverse SDE Health]")
    print(f"  Diverged (‖θ‖>100):  {div.get('divergence_ratio_1e2', 0):.2%}")
    print(f"  Max ‖θ‖:             {div.get('max_norm', 0):.2f}")
    print(f"  NaN/Inf ratio:       {div.get('nan_ratio', 0):.2%} / {div.get('inf_ratio', 0):.2%}")

    gs = all_metrics.get('gauss_stability', {})
    print(f"\n[GAUSS Stability]")
    print(f"  Eigenvalue Correction Ratio: {gs.get('eigenvalue_correction_ratio', 0):.2%}")
    print(f"  Mean Condition Number:       {gs.get('mean_condition_number', 0):.1f}")

    timing = all_metrics.get('timing', {})
    print(f"\n[Timing]")
    for stage, info in timing.items():
        print(f"  {stage:20s} {info['seconds']:8.1f}s  ({info['fraction']:.1%})")

    pred = all_metrics.get('predictive', {})
    print(f"\n[Predictive Check]")
    print(f"  Mean Traj. RMSE:     {pred.get('mean_trajectory_rmse', 0):.4f}")
    print(f"  Pred. Coverage 90%:  {pred.get('predictive_coverage_90', 0):.2%}")

    print(f"\n{header}\n")
