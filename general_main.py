import jax
import jax.numpy as jnp
from src.factory import get_simulator
from src.local_score_net import LocalScoreNet
from src.reverse import Diffuser, EulerMaruyama
from src.composition_score_fn import GAUSSScoreFn

# 
def train(key, model, sim, proposal_fn, train_config):
    """범용 훈련 루프: 시뮬레이터의 차원을 자동으로 감지합니다."""
    # 시뮬레이터에서 차원 정보 추출 [cite: 80, 113]
    theta_dim = getattr(sim, 'theta_dim', 4) # 기본값 4 (LV 기준)
    data_dim = getattr(sim, 'dim', 2)
    
    # 설정값 로드
    num_steps = train_config.get('num_steps', 10000)
    batch_size = train_config.get('batch_size', 128)
    
    optimizer = optax.adamw(learning_rate=train_config.get('lr', 5e-4))
    
    # 동적 초기화 [cite: 928]
    init_theta = jnp.ones((1, theta_dim))
    init_x = jnp.ones((1, 2, data_dim)) # (B, T, d_x)
    params = model.init(key, init_x, init_theta, jnp.ones((1,)))['params']
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, k, theta, x_batch):
        # 단일 전이(x_t, x_next)에 대한 손실 계산 [cite: 131, 133]
        loss, grads = jax.value_and_grad(denoising_score_matching_loss)(
            params, model, k, theta, x_batch[:, 0], x_batch[:, 1], SDE()
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    for i in range(num_steps):
        key, k_theta, k_prop, k_sim = jax.random.split(key, 4)
        
        # 시뮬레이터별 사전 분포/제안 분포 호출 [cite: 113]
        theta = sim.prior(k_theta, batch_size) if hasattr(sim, 'prior') else sim.prior_sampler(k_theta, batch_size)
        _, x_t = proposal_fn(k_prop, batch_size)
        x_next = sim.transition(k_sim, x_t, theta)
        
        x_batch = jnp.stack([x_t, x_next], axis=1)
        params, opt_state, loss = step(params, opt_state, key, theta, x_batch)
        
    return params

def main(sim_name):
    key = jax.random.PRNGKey(42)
    
    # 1. 시뮬레이터 선택 (자유롭게 변경 가능)
    sim, proposal = get_simulator(sim_name, config['sim'])
    
    # 2. 모델 설정 (시뮬레이터 차원에 맞춰 자동 조절될 수 있도록 config 관리)
    model = LocalScoreNet(config=config['model'])
    
    # 3. 알고리즘 1: 로컬 스코어 학습 (단일 전이 학습) [cite: 58, 120]
    trained_params = train(key, model, sim, proposal, config['train'])
    
    # 4. 관측 데이터 생성 (테스트용) [cite: 80, 86]
    # 실제 환경에서는 x_obs가 외부 데이터로 주어짐
    x_obs = generate_test_data(sim, key) 

    # 5. 알고리즘 2: 스코어 결합 및 추론 (전체 시계열 처리) [cite: 60, 115, 143]
    samples = infer(key, model, trained_params, sim, x_obs)
    return samples

if __name__ == "__main__":
    # 원하는 시뮬레이터 이름을 인자로 넣어 실행
    # 예: 'SIR', 'GaussianRandomWalk', 'KolmogorovFlow'
    samples = main('SIR')