import matplotlib.pyplot as plt

def plot_posterior_samples(samples, theta_true, param_names=None, save_path=None):
    """
    사후 분포 샘플과 실제 파라미터 값을 비교하는 히스토그램 시각화
    """
    if param_names is None:
        param_names = ['alpha', 'beta', 'delta', 'gamma']
        
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    theta_true_flat = theta_true.flatten()

    for i in range(4):
        # 예측된 사후 분포 (Posterior)
        axes[i].hist(samples[:, i], bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        # 실제 파라미터 값 (Ground Truth)
        axes[i].axvline(theta_true_flat[i], color='red', linestyle='dashed', linewidth=2, label='True Value')
        
        axes[i].set_title(f'Posterior of {param_names[i]}')
        axes[i].legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()