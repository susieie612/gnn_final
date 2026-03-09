import jax
import jax.numpy as jnp
from jax import random

import math

"""
Benchmarks:

Gaussian Random Walk
"""

class GaussianRandomWalk: 
    def __init__(self, dim = 1, alpha = 0.9, sigma = 1.0): 
        self.dim = dim 
        self.alpha = alpha 
        self.sigma = sigma 
        
    def prior(self, key, batch): 
        return random.uniform(key,(batch, self.dim), minval = -3.0, maxval = 3.0) 
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch, self.dim)) 
        return key, x
    
    def transition(self,key, x, theta): 
        eps = self.sigma * random.normal(key, x.shape) 
        return self.alpha * x + theta + eps 

"""
Mixture Random Walk
"""

class MixtureRandomWalk:
    def __init__(self, dim = 5, sigma = 1.0):
        self.dim = dim
        self.sigma = sigma
        
    def prior(self, key, batch): 
        return random.uniform(key,(batch, self.dim), minval = -3.0, maxval = 3.0) 
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch, self.dim)) 
        return key, x
    
    def transition(self, key, x, theta):
        u_key, eps_key = random.split(key)
        
        u = random.choice(u_key, jnp.array([-1.0, 1.0]), shape = (x.shape[0],1))
        eps = self.sigma * random.normal(eps_key, x.shape)
        return x + u*theta + eps
    
"""
Periodic SDE
"""

class PeriodicSDE:
    def __init__(self, dt = 0.05, sigma = 0.5):
        self.dt = dt
        self.sigma = sigma
        self.dim = 2
    
    def prior(self, key, batch):
        return random.uniform(key, (batch,1), minval = -3.0, maxval = 3.0)
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch,2))
        return key,x
    
    def transition(self,key,x,theta):
        omega = theta[:,0]
        
        zeros = jnp.zeros_like(omega)
        A = jnp.stack([
            jnp.stack([zeros, -omega], axis = 1),
            jnp.stack([omega, zeros], axis=1)], axis = 1)
        drift = jnp.matmul(A,x[...,None]).squeeze(-1)
        noise = self.sigma * jnp.sqrt(self.dt) * random.normal(key, x.shape)
        return  x + drift * self.dt + noise



"""
Linear SDE
"""

class LinearSDE:
    def __init__(self, dt = 0.05, dim = 3, theta_dim = 18):
        self.dt = dt
        self.dim = dim
        self.theta_dim = theta_dim
        
    def prior(self, key, batch):
        return jnp.random.uniform(-1,1, size = 18)
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch,3))
        return key,x
    
    def unpack(self, theta):
        A = theta[:,:9].reshape(-1,3,3)
        B = theta[:,9:].reshape(-1,3,3)
        return A,B
    
    def transition(self,key,x, theta):
        A,B = self.unpack(theta)
        drift = jnp.matmul(A, x[...,None]).squeeze(-1)
        eps = random.normal(key,x.shape)
        diff = jnp.matmul(B,eps[...,None]).squeeze(-1)
        
        return x + drift * self.dt + diff * jnp.sqrt(self.dt)
    
"""
Double Well SDE
"""

class DoubleWellSDE:
    def __init__(self,dt=0.01, sigma = 0.5, dim = 1):
        self.dt = dt
        self.sigma = sigma
        self.dim = dim
    
    def prior(self, key , batch):
        k1,k2 = random.split(key)
        t1 = random.uniform(k1, (batch,1), minval = -2.0, maxval=2.0)
        t2 = random.uniform(k2, (batch,1), minval = -2.0, maxval = 0.0)
        return jnp.concatenate([t1,t2], axis = 1)
    
    def proposal(self, key, batch):
        x = random.normal(key, (batch,1))
        return key,x
    
    def transition(self,key,x,theta):
        t1 = theta[:,0:1]
        t2=theta[:,1:2]
        drift = t1*x + t2*(x**3)
        noise = self.sigma * jnp.sqrt(self.dt) * random.normal(key,x.shape)
        
        return x + drift * self.dt + noise


"""
Simulate one long trajectory instead of arbitrary states and 
resample states visited by the simulator
"""

def lv_prior_predictive_proposal(sim, key, n_steps = 5000):
    theta_key, x0_key, sim_key = random.split(key,3)
    
    theta = sim.prior_sampler(theta_key,1)
    
    x = random.uniform(x0_key, (1,2), minval = 0.0, maxval = 10.0)
    
    states = []
    
    #Simulate trajectory
    for step in range(n_steps):
        sim_key, step_key = random.split(sim_key)
        x = sim.transition(step_key, x, theta)
        states.append(x)
        
    states = jnp.concatenate(states, axis = 0)
    
    def proposal(key, batch):
        idx = random.randint(key, (batch,), 0, states.shape[0])
        return key, states[idx]
    
    return proposal

# LV trajectories

def lv_traj(key, theta, x0, T):
    states = [x0]
    x = x0
    
    for t in range(T):
        key,subkey = random.split(key)
        x = LotkaVolterra.transition(subkey, x, theta)
        states.append(x)
    
    return jnp.stack(states)

"""
SIR 

S = susceptible
I = infected
R = recovered
"""

class SIR:
    def __init__(self, sigma = 0.02, dt = 0.01, eps = 1e-6):
        self.sigma = sigma
        self.dt = dt
        self.eps = eps

    def transition(self, key, x_t, theta):
        beta, gamma = theta.T
        S,I,R = x_t.T
        
        #Deterministic ODE
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        dR = gamma * I
        
        k1, k2, k3 = random.split(key, 3)
        eps_S = random.normal(k1, S.shape)
        eps_I = random.normal(k2, I.shape)
        eps_R = random.normal(k3, R.shape)
        
        diffusion_S = self.sigma * S
        diffusion_I = self.sigma * I
        diffusion_R = self.sigma * R
        
        # Euler-Maruyama step
        S_next = S + dS * self.dt + diffusion_S * jnp.sqrt(self.dt) * eps_S
        I_next = I + dI * self.dt + diffusion_I * jnp.sqrt(self.dt) * eps_I
        R_next = R + dR * self.dt + diffusion_R * jnp.sqrt(self.dt) * eps_R
        
        # to enforce positivity
        S_next = jnp.maximum(S_next, self.eps)
        I_next = jnp.maximum(I_next, self.eps)
        R_next = jnp.maximum(R_next, self.eps)
        
        return jnp.stack([S_next, I_next, R_next], axis = 1)

    def uniform_proposal( self, key, batch):
        key, subkey = random.split(key)
        S = random.uniform(subkey, (batch,), minval = 0.5, maxval = 1.0)
        I = random.uniform(subkey, (batch,), minval = 0.0, maxval = 0.5)
        R = jnp.zeros(batch)
        
        x = jnp.stack([S,I,R], axis = 1)
        
        return key, x

    def prior_sampler(self,key, batch):
        x = random.normal(key, (batch,2))
        return jnp.exp(x)
    
    """
Kolmogorov Flow 

Use Novier Stokes equations
"""

class KolmogorovFlow:
    def __init__(self, dt = 0.01, sigma = 5e-3, eps = 1e-8, N = 32, L = 2 * math.pi):
        self.dt = dt
        self.sigma = sigma
        self.eps = eps
        
        self.N = N
        self.L = L
        
        k = jnp.fft.fftfreq(N, d = L/(2 * math.pi * N))
        kx, ky = jnp.meshgrid(k,k, indexing="ij")
        self.kx=kx
        self.ky=ky
        self.k2=kx**2 + ky**2
        self.k2 = self.k2.at[0, 0].set(1.0)
        y=jnp.linspace(0, L, N)
        self.forcing_pattern = jnp.sin(y) [None, :].repeat(N, axis=0)
        
    def prior(self, key, batch):
        key1, key2 = random.split(key)
        Re=random.uniform(key1, (batch, 1), minval=0.8, maxval=1.2)
        rho=random.uniform(key2, (batch, 1),minval=0.5, maxval=2.0)
        theta=jnp.concatenate( [Re, rho], axis=1)
        return theta 
    
    
    def x0(self, key, batch):
        return 0.1 * random.normal(key, (batch, self.N, self.N))
 
    def transition(self, key, omega, theta):
        batch = omega.shape[0]
        Re = theta[:, 0][:, None, None]
        rho = theta[:, 1][:, None, None]
        
        omega_hat = jnp.fft.fft2(omega)
        
        psi_hat = - omega_hat/self.k2
        
        u_hat = 1j * self.ky * psi_hat
        v_hat = -1j * self.kx * psi_hat
        
        u = jnp.real(jnp.fft.ifft2(u_hat))
        v = jnp.real(jnp.fft.ifft2(v_hat))
        
        #Gradients
        domega_dx = jnp.real(jnp.fft.ifft2(1j * self.kx * omega_hat))
        domega_dy = jnp.real(jnp.fft.ifft2(1j * self.ky * omega_hat))
        
        nonlinear = u * domega_dx + v * domega_dy
        
        laplacian = jnp.real(jnp.fft.ifft2(-self.k2 * omega_hat))
        
        forcing = rho * self.forcing_pattern
        
        #Euler update
        omega_next = omega + self.dt * (
            -nonlinear + (1.0/Re) * laplacian + forcing
        )
        
        noise = self.sigma * jnp.sqrt(self.dt) * random.normal(key, omega.shape)
        omega_next = omega_next + noise
        
        return omega_next

def kf_proposal(sim):
        def proposal(key, batch):
            return key, sim.x0(key, batch)
        return proposal

# Generate Dataset

def generate_transition_dataset(key, N, prior_fn, proposal_fn, transition_fn):
    key_theta, key_x, key_step = random.split(key, 3)
    
    #sample parameters
    theta = prior_fn(key_theta, N)
    
    #sample state
    _, x_t = proposal_fn(key_x, N)
    
    x_tp1 = transition_fn(key_step, x_t, theta)
    
    return theta, x_t, x_tp1

# Training Datasets

key = random.PRNGKey(0)
key_lv, key_sir, key_kf, key_grw, key_mrw, key_p, key_l, key_dw = random.split(key,8)


#Lotka Volterra
lv_sim = LotkaVolterra()
lv_proposal = lv_prior_predictive_proposal(lv_sim,key_lv)

theta_lv, x_t_lv, x_tp1_lv = generate_transition_dataset(
    key_lv, 50000,
    lv_sim.prior_sampler,
    lv_proposal,
    lv_sim.transition
)

# SIR
sir_sim = SIR()
theta_sir, x_t_sir, x_tp1_sir = generate_transition_dataset(
    key_sir, 50000,
    sir_sim.prior_sampler,
    sir_sim.uniform_proposal,
    sir_sim.transition
    
)

#Kolmogorov Flow
kf_sim = KolmogorovFlow(N=32)
theta_kf, omega_t_kf, omega_tp1_kf = generate_transition_dataset(
    key_kf, 20000,
    kf_sim.prior,
    kf_proposal(kf_sim),
    kf_sim.transition
)


# Benchmarks training datasets

grw = GaussianRandomWalk(dim = 3)

theta_grw, x_t_grw, x_tp1_grw = generate_transition_dataset(
    key_grw, 50000, grw.prior, grw.proposal, grw.transition)

mrw = MixtureRandomWalk(dime = 3)

theta_mrw, x_t_mrw, x_tp1_mrw = generate_transition_dataset(
    key_mrw, 50000, mrw.prior, mrw.proposal, mrw.transition
)

periodic = PeriodicSDE()

theta_p, x_t_p, x_tp1_p = generate_transition_dataset(
    key_p, 50000, periodic.prior, periodic.proposal, periodic.transition
)

linear = LinearSDE()

theta_l, x_t_l, x_tp1_l = generate_transition_dataset(
    key_l, 50000, linear.prior, linear.proposal, linear.transition
)

dw = DoubleWellSDE()

theta_dw, x_t_dw, x_tp1_l = generate_transition_dataset(
    key_dw, 50000, dw.prior, dw.proposal, dw.transition
)