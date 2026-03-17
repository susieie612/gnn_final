import jax
import jax.numpy as jnp
from jax import random
import math

class GaussianRandomWalk: 
    def __init__(self, dim=1, alpha=0.9, sigma=1.0): 
        self.dim = dim 
        self.alpha = alpha 
        self.sigma = sigma 
        self.d_x = dim      
        self.d_theta = dim
        
    def prior(self, key, batch): 
        return random.uniform(key, (batch, self.dim), minval=-3.0, maxval=3.0) 
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch, self.dim)) 
        return key, x
    
    def transition(self, key, x, theta): 
        eps = self.sigma * random.normal(key, x.shape) 
        return self.alpha * x + theta + eps 

class MixtureRandomWalk:
    def __init__(self, dim=5, sigma=1.0):
        self.dim = dim
        self.sigma = sigma
        self.d_x = dim      
        self.d_theta = dim
        
    def prior(self, key, batch): 
        return random.uniform(key, (batch, self.dim), minval=-3.0, maxval=3.0) 
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch, self.dim)) 
        return key, x
    
    def transition(self, key, x, theta):
        u_key, eps_key = random.split(key)
        u = random.choice(u_key, jnp.array([-1.0, 1.0]), shape=(x.shape[0], 1))
        eps = self.sigma * random.normal(eps_key, x.shape)
        return x + u * theta + eps

class PeriodicSDE:
    def __init__(self, dt=0.05, sigma=0.5):
        self.dt = dt
        self.sigma = sigma
        self.dim = dim
        self.d_x = 2
        self.d_theta = 2
    
    def prior(self, key, batch):
        return random.uniform(key, (batch, 1), minval=-3.0, maxval=3.0)
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch, 2))
        return key, x
    
    def transition(self, key, x, theta):
        omega = theta[:, 0]
        zeros = jnp.zeros_like(omega)
        A = jnp.stack([
            jnp.stack([zeros, -omega], axis=1),
            jnp.stack([omega, zeros], axis=1)], axis=1)
        drift = jnp.matmul(A, x[..., None]).squeeze(-1)
        noise = self.sigma * jnp.sqrt(self.dt) * random.normal(key, x.shape)
        return x + drift * self.dt + noise

class LinearSDE:
    def __init__(self, dt=0.05, dim=3, theta_dim=18):
        self.dt = dt
        self.dim = dim
        self.d_x = dim
        self.d_theta = theta_dim
        
    def prior(self, key, batch):
        return random.uniform(key, (batch, self.theta_dim), minval=-1.0, maxval=1.0)
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch, self.dim))
        return key, x
    
    def unpack(self, theta):
        A = theta[:, :9].reshape(-1, 3, 3)
        B = theta[:, 9:].reshape(-1, 3, 3)
        return A, B
    
    def transition(self, key, x, theta):
        A, B = self.unpack(theta)
        drift = jnp.matmul(A, x[..., None]).squeeze(-1)
        eps = random.normal(key, x.shape)
        diff = jnp.matmul(B, eps[..., None]).squeeze(-1)
        return x + drift * self.dt + diff * jnp.sqrt(self.dt)

class DoubleWellSDE:
    def __init__(self, dt=0.01, sigma=0.5, dim=1):
        self.dt = dt
        self.sigma = sigma
        self.d_x = 4
        self.d_theta = 2
        self.dim = dim
    
    def prior(self, key, batch):
        k1, k2 = random.split(key)
        t1 = random.uniform(k1, (batch, 1), minval=-2.0, maxval=2.0)
        t2 = random.uniform(k2, (batch, 1), minval=-2.0, maxval=0.0)
        return jnp.concatenate([t1, t2], axis=1)
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch, self.dim))
        return key, x
    
    def transition(self, key, x, theta):
        t1 = theta[:, 0:1]
        t2 = theta[:, 1:2]
        drift = t1 * x + t2 * (x**3)
        noise = self.sigma * jnp.sqrt(self.dt) * random.normal(key, x.shape)
        return x + drift * self.dt + noise

class LotkaVolterra:
    def __init__(self, sigma=0.05, dt=0.1, eps=1e-6):
        self.sigma = sigma
        self.dt = dt
        self.eps = eps
        self.d_x = 2      # [x, y]
        self.d_theta = 4  # [alpha, beta, delta, gamma]
        
    def transition(self, key, x_t, theta):
        alpha, beta, delta, gamma = theta.T
        x, y = x_t.T
        
        dx = alpha * x - beta * x * y
        dy = -gamma * y + delta * x * y
        
        k1, k2 = random.split(key)
        noise_x = random.normal(k1, x.shape)
        noise_y = random.normal(k2, y.shape)
        diffusion = self.sigma * x * y
        
        x_next = x + dx * self.dt + diffusion * jnp.sqrt(self.dt) * noise_x
        y_next = y + dy * self.dt + diffusion * jnp.sqrt(self.dt) * noise_y
        
        return jnp.stack([jnp.maximum(x_next, self.eps), jnp.maximum(y_next, self.eps)], axis=1)
    
    def prior(self, key, batch):
        raw = random.normal(key, (batch, 4))
        return raw # jnp.exp(raw)

    def proposal(self, key, batch):
        key, subkey = random.split(key)
        x = random.uniform(subkey, (batch, 2), minval=0.0, maxval=10.0)
        return key, x

class SIR:
    def __init__(self, sigma=0.02, dt=0.01, eps=1e-6):
        self.sigma = sigma
        self.dt = dt
        self.eps = eps
        self.d_x = 3      # [S, I, R]
        self.d_theta = 2  # [beta, gamma]

    def transition(self, key, x_t, theta):
        beta, gamma = theta.T
        S, I, R = x_t.T
        
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        
        k1, k2, k3 = random.split(key, 3)
        eps_S = random.normal(k1, S.shape)
        eps_I = random.normal(k2, I.shape)
        eps_R = random.normal(k3, R.shape)
        
        S_next = S + dS * self.dt + (self.sigma * S) * jnp.sqrt(self.dt) * eps_S
        I_next = I + dI * self.dt + (self.sigma * I) * jnp.sqrt(self.dt) * eps_I
        R_next = R + dR * self.dt + (self.sigma * R) * jnp.sqrt(self.dt) * eps_R
        
        return jnp.stack([jnp.maximum(v, self.eps) for v in [S_next, I_next, R_next]], axis=1)

    def prior(self, key, batch):
        x = random.normal(key, (batch, 2))
        return x #jnp.exp(x)

    def proposal(self, key, batch):
        key, subkey = random.split(key)
        S = random.uniform(subkey, (batch,), minval=0.5, maxval=1.0)
        I = random.uniform(subkey, (batch,), minval=0.0, maxval=0.5)
        R = jnp.zeros(batch)
        return key, jnp.stack([S, I, R], axis=1)

class KolmogorovFlow:
    def __init__(self, dt=0.01, sigma=5e-3, eps=1e-8, N=32, L=2 * math.pi):
        self.dt, self.sigma, self.eps = dt, sigma, eps
        self.N, self.L = N, L
        k = jnp.fft.fftfreq(N, d=L/(2 * math.pi * N))
        kx, ky = jnp.meshgrid(k, k, indexing="ij")
        self.kx, self.ky = kx, ky
        self.k2 = kx**2 + ky**2
        self.k2 = self.k2.at[0, 0].set(1.0)
        y = jnp.linspace(0, L, N)
        self.forcing_pattern = jnp.sin(y)[None, :].repeat(N, axis=0)
        self.d_x = 4096
        self.d_theta = 2
        
    def prior(self, key, batch):
        key1, key2 = random.split(key)
        Re = random.uniform(key1, (batch, 1), minval=0.8, maxval=1.2)
        rho = random.uniform(key2, (batch, 1), minval=0.5, maxval=2.0)
        return jnp.concatenate([Re, rho], axis=1)
    
    def x0(self, key, batch):
        return 0.1 * random.normal(key, (batch, self.N, self.N))
 
    def transition(self, key, omega, theta):
        Re, rho = theta[:, 0][:, None, None], theta[:, 1][:, None, None]
        omega_hat = jnp.fft.fft2(omega)
        psi_hat = - omega_hat / self.k2
        u = jnp.real(jnp.fft.ifft2(1j * self.ky * psi_hat))
        v = jnp.real(jnp.fft.ifft2(-1j * self.kx * psi_hat))
        domega_dx = jnp.real(jnp.fft.ifft2(1j * self.kx * omega_hat))
        domega_dy = jnp.real(jnp.fft.ifft2(1j * self.ky * omega_hat))
        
        nonlinear = u * domega_dx + v * domega_dy
        laplacian = jnp.real(jnp.fft.ifft2(-self.k2 * omega_hat))
        forcing = rho * self.forcing_pattern
        
        omega_next = omega + self.dt * (-nonlinear + (1.0/Re) * laplacian + forcing)
        return omega_next + self.sigma * jnp.sqrt(self.dt) * random.normal(key, omega.shape)

# --- Helper functions for Proposals ---

def lv_prior_predictive_proposal(sim, key, n_steps=5000):
    theta_key, x0_key, sim_key = random.split(key, 3)
    theta_raw = sim.prior(theta_key, 1)
    theta = jnp.exp(theta_raw)
    
    x = random.uniform(x0_key, (1, 2), minval=0.0, maxval=10.0)
    
    states = []
    for _ in range(n_steps):
        sim_key, step_key = random.split(sim_key)
        x = sim.transition(step_key, x, theta)
        states.append(x)
    
    states = jnp.concatenate(states, axis=0)
    return lambda k, b: (k, states[random.randint(k, (b,), 0, states.shape[0])])

def kf_proposal(sim):
    return lambda k, b: (k, sim.x0(k, b))

# def lv_traj(key, sim, theta, x0, T):
#     states = [x0]
#     x = x0
#     for _ in range(T):
#         key, subkey = random.split(key)
#         x = sim.transition(subkey, x, theta)
#         states.append(x)
#     return jnp.stack(states)

def lv_traj(key, sim, theta, x0, T):
    states = [x0]
    x = x0
    for _ in range(T):
        key, subkey = random.split(key)
        x = sim.transition(subkey, x, theta)
        states.append(x)
    
    return jnp.stack(states).squeeze(axis=1)