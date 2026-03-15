from src.simulators import (
    LotkaVolterra, SIR, GaussianRandomWalk, MixtureRandomWalk, 
    PeriodicSDE, LinearSDE, DoubleWellSDE, KolmogorovFlow,
    lv_prior_predictive_proposal, kf_proposal
)

def get_simulator(name, config):
    """설정값에 따라 시뮬레이터와 제안 분포 함수를 반환합니다."""
    sim_params = config.get('sim_params', {})
    
    if name == 'LotkaVolterra':
        sim = LotkaVolterra(**sim_params)
        proposal_fn = lv_prior_predictive_proposal(sim, jax.random.PRNGKey(0))
    elif name == 'SIR':
        sim = SIR(**sim_params)
        proposal_fn = sim.uniform_proposal
    elif name == 'GaussianRandomWalk':
        sim = GaussianRandomWalk(**sim_params)
        proposal_fn = sim.proposal
    elif name == 'KolmogorovFlow':
        sim = KolmogorovFlow(**sim_params)
        proposal_fn = kf_proposal(sim)
    # 필요한 다른 시뮬레이터들을 elif로 추가
    else:
        raise ValueError(f"Unknown simulator: {name}")
        
    return sim, proposal_fn