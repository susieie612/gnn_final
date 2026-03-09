import jax
import jax.numpy as jnp
import optax
from src.local_score_net import LocalScoreNet
from src.loss import denoising_score_matching_loss
from src.reverse import LocalDiffuser
from src.kernels import EulerMaruyama
from src.composition_score_fn import GAUSSScoreFn

# --- 0. 환경 설정 및 SDE 정의 ---
class SDE:
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.beta_min = beta_min [cite: 939]
        self.beta_max = beta_max [cite: 939]
        self.T_min = 1e-4
        self.T_max = 1.0

    def std(self, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return jnp.sqrt(1.0 - jnp.exp(2.0 * log_mean_coeff)) [cite: 940]

    def mean(self, t, theta):
        log_mean = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        return jnp.exp(log_mean) * theta [cite: 940]

# --- 1. ALGORITHM 1: TRAINING ---
def run_training(key, model, simulator_fn, prior, proposal, num_steps=5000):
    sde = SDE()
    optimizer = optax.adamw(learning_rate=5e-4) [cite: 945]
    
    # 모델 초기화
    init_key, key = jax.random.split(key)
    params = model.init(init_key, jnp.ones((1, 2, 1)), jnp.ones((1, 2)), jnp.ones((1,)))['params'] [cite: 113]
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(params, opt_state, k, theta, x_batch):
        # Eq. 4 기반 손실 계산 및 그레이디언트 업데이트 [cite: 131, 133]
        loss, grads = jax.value_and_grad(denoising_score_matching_loss)(
            params, model, k, theta, x_batch[:, 0], x_batch[:, 1], sde
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("Algorithm 1: Training local score net...")
    for step in range(num_steps):
        key, subkey = jax.random.split(key)
        
        # 데이터 샘플링 (Prior & Proposal 사용) [cite: 113]
        theta = prior.sample(subkey, (128,))
        x_t = proposal.sample(subkey, (128,))
        x_next = simulator_fn(x_t, theta)
        x_batch = jnp.stack([x_t, x_next], axis=1) # (B, 2, d_x)
        
        params, opt_state, loss = update_step(params, opt_state, key, theta, x_batch)
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
            
    return params

# --- 2. ALGORITHM 2: EVALUATION (INFERENCE) ---
def run_inference(key, model, params, sde, prior, x_obs, marginal_prior_fn):
    grid = jnp.linspace(1e-3, 1.0, 100)
    
    # 로컬 정밀도 추정을 위한 내부 Diffuser 설정 [cite: 115, 121]
    # EM 커널은 LocalScoreNet을 사용하여 각 전이의 스코어를 계산합니다.
    local_kernel = EulerMaruyama(model, params, sde)
    local_diffuser = LocalDiffuser(local_kernel, grid, (prior.dim,), sde)
    
    # GAUSS 기법을 이용한 조합 스코어 함수 (식 150, 151) [cite: 150, 151]
    gauss_score_fn = GAUSSScoreFn(
        score_net=model,
        params=params,
        sde=sde,
        prior=prior,
        diffuser=local_diffuser,
        marginal_prior_fn=marginal_prior_fn,
        num_samples=500 # 논문 권장 샘플 수 [cite: 818]
    )

    # 글로벌 사후 분포 샘플링용 최종 커널 
    global_kernel = EulerMaruyama(gauss_score_fn, params, sde)
    global_diffuser = LocalDiffuser(global_kernel, grid, (prior.dim,), sde)
    
    print("Algorithm 2: Sampling global posterior via GAUSS composition...")
    # 전체 관측 타임 시리즈 x_obs를 조건으로 샘플링 [cite: 96, 121]
    final_samples = jax.vmap(lambda k: global_diffuser.sample(k, x_obs, a_start=1.0))(
        jax.random.split(key, 100) # 100개의 사후 분포 샘플 생성
    )
    
    return final_samples

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    main_key = jax.random.PRNGKey(0)
    fnse_model = LocalScoreNet(d_teta=2)
    sde_inst = SDE()

    # Step 1: 학습 [cite: 112]
    trained_params = run_training(main_key, fnse_model, my_simulator, my_prior, my_proposal)

    # Step 2: 추론 (관측 데이터 기반) [cite: 115]
    # x_obs_target은 (T+1, d_x) 형태의 실제 관측 데이터입니다.
    samples = run_inference(main_key, fnse_model, trained_params, sde_inst, my_prior, x_obs_target, my_marginal_prior)
    
    print(f"Final Posterior Samples Shape: {samples.shape}")