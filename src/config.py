# src/config.py
config = {
    "active_sim": "GaussianRandomWalk", ## change this part to use different simulator
    
    "model": {
        "hidden_dim": 128,
        "d_theta": 4,
        "d_x": 2
    },
    "train": {
        "learning_rate": 5e-4,
        "num_steps": 10000,
        "batch_size": 128
    },
    "sim": {
        "dt": 0.1,
        "sigma": 0.05
    },

    "sim_settings": {
        "GaussianRandomWalk": {
            "name": "GaussianRandomWalk",
            "params": {"dim": 1, "alpha": 0.9, "sigma": 1.0},
            "target": {"theta_true": [0.5], "x0_true": [0.0], "time_steps": 100}
        },
        "MixtureRandomWalk": {
            "name": "MixtureRandomWalk",
            "params": {"dim": 5, "sigma": 1.0},
            "target": {"theta_true": [0.5, 0.5, 0.5, 0.5, 0.5], "x0_true": [0.0, 0.0, 0.0, 0.0, 0.0], "time_steps": 10}
        },
        "PeriodicSDE": { ## currently not working
            "name": "PeriodicSDE",
            "params": {"dt": 0.05, "sigma": 0.5},
            "target": {"theta_true": [2.0, 0.0], "x0_true": [1.0, 0.0], "time_steps": 5}
        },
        "LinearSDE": {
            "name": "LinearSDE",
            "params": {"dt": 0.05, "dim": 3, "theta_dim": 18},
            "target": {
                "theta_true": [0.1] * 18, 
                "x0_true": [1.0, 0.5, -0.5], 
                "time_steps": 20
            }
        },
        "DoubleWellSDE": {
            "name": "DoubleWellSDE",
            "params": {"dt": 0.01, "sigma": 0.5, "dim": 1},
            "target": {"theta_true": [1.0, -0.5], "x0_true": [0.1], "time_steps": 5}
        },
        "LotkaVolterra": {
            "name": "LotkaVolterra",
            "params": {"sigma": 0.05, "dt": 0.1},
            "target": {"theta_true": [0.6, 0.025, 0.8, 0.025], "x0_true": [10.0, 5.0], "time_steps": 5}
        },
        "SIR": {
            "name": "SIR",
            "params": {"sigma": 0.02, "dt": 0.01},
            "target": {"theta_true": [0.2, 0.1], "x0_true": [0.99, 0.01, 0.0], "time_steps": 50}
        },
        "KolmogorovFlow": {
            "name": "KolmogorovFlow",
            "params": {"dt": 0.01, "sigma": 5e-3, "N": 32},
            "target": {"theta_true": [1.0, 1.0], "x0_true": None, "time_steps": 50} # x0는 x0(key, batch) 메서드로 생성 권장
        }
    },
}