# src/config.py
config = {
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
    }
}