import jax
import jax.numpy as jnp


class GAUSSScoreFn:
    """Based on section 3.2.2 GAUSS model"""
    def __init__(self, score_net, params, sde, prior, diffuser, marginal_prior_fn, num_samples=500):
        self.score_net = score_net # LocalScoreNet 
        self.params = params
        self.sde = sde
        self.prior = prior
        self.diffuser = diffuser # reverse.py
        self.marginal_prior_fn = marginal_prior_fn
        self.num_samples = num_samples

    def __call__(self, a, theta_a, x_0_T, ensure_pd = True, **kwargs):
        """
        Args:
        a: diffusion time
        theta_a: noisy parameter (d_theta, )
        x_0_T: total observations
        """
        print(f'x_0_T shape: {x_0_T.shape}')
        # make sure that the shape is  (B, ...)
        if x_0_T.ndim != 3:
            x_0_T = x_0_T.reshape(-1, x_0_T.shape[-2], x_0_T.shape[-1])

        # elif x_0_T.ndim == 2:
        #     x_0_T = x_0_T[jnp.newaxis, ...] # (B, T, d_x)
        print(f'x_0_T shape after reshaping: {x_0_T.shape}')
        B, T, _ = x_0_T.shape
        num_transitions = T - 1

        # 1. calculate the local score (s_phi)
        theta_in = theta_a[jnp.newaxis, ...] if theta_a.ndim == 1 else theta_a # (B, d_tehta)
        a_in = jnp.atleast_1d(a) # (B, )

        local_scores = self.score_net.apply({'params': self.params}, x_0_T, theta_in, a_in) # (B, T-1, d_theta)
        print(f'network output shape : {local_scores.shape}')
        local_scores = jnp.squeeze(local_scores, axis=0) 
        print(f'network output shape after reshaping : {local_scores.shape}')
        # local_scores = local_scores[0] # (T-1, d_theta)

        # 2. Calculate the inverse covarainces
        # \Sigma_a^-1  --> precision of p(\theta | theta_a)
        sigma_a_inv = self.get_prior_precision(a) # (d_tehta, d_theta)
        # \Sigma_a,t,t+1^-1 --> precision of local posterior p(\theta | theta_a, x^t,t+1)
        sigma_a_t_inv = self.estimate_local_precision(a, theta_a, x_0_T) # (T-1, d_tehta, d_tehta)
        
        # 3. lambda_a = Sum (\simga_a,t,t+1 ^-1 + (1-T) \sigma_a^-1)
        lambda_a = jnp.sum(sigma_a_t_inv, axis=0) + (1 - num_transitions) * sigma_a_inv
        
        # makse sure lambda_a is positive dfinite using eigenvalue decomposition (appendix B. 3) for numerical stability
        if ensure_pd is True:
            eps = 1e-5
            eigenvalues, eigenvectors = jnp.linalg.eigh(lambda_a)
            adjusted_ev = jnp.maximum(eigenvalues, eps)
            lambda_a = eigenvectors @ jnp.diag(adjusted_ev) @ eigenvectors.T 

        # 4. compose global score
        # 1st term
        weighted_local_sum = jnp.sum(jax.vmap(lambda prec, score: prec @ score)(sigma_a_t_inv, local_scores), axis=0)
        # 2nd term
        prior_score = self.marginal_prior_fn(a, jnp.squeeze(theta_a)) # p(theta_a)
        weighted_prior_term = (1 - num_transitions) * (sigma_a_inv @ prior_score)

        global_score = jnp.linalg.solve(lambda_a, weighted_local_sum + weighted_prior_term)

        return global_score

    def apply(self, others, x_0_T, theta_a, a):
        """ to match the EulserMaruyama interface (reverse.py)"""
        theta_val = jnp.squeeze(theta_a)
        a_val = jnp.squeeze(a)
        return self.__call__(a_val, theta_val, x_0_T)
    

    def get_prior_precision(self, a): ## TODO: how do i get this
    
        return jnp.eye(self.prior.dim) / ( self.sde.std(a)**2 )
    
    def estimate_local_precision(self, a, theta_a, x_0_T):
        """
        estimate the local posterior by sampling using the diffuser
        """
        x_starts = x_0_T[0, :-1, :]
        x_ends = x_0_T[0, 1:, :]
        x_pairs = jnp.concatenate([x_starts, x_ends], axis=-1) # (num_transitions, d_x_pair)

        def single_transition_precision(x_pair):
            keys = jax.random.split(jax.random.PRNGKey(0), self.num_samples)

            samples = jax.vmap(lambda k: self.diffuser.sample(k, x_pair))(keys)

            cov = jnp.cov(samples, rowvar=False)
            return jnp.linalg(cov + 1e-6 * jnp.eye(cov.shape[0]))

        return jax.vmap(single_transition_precision(x_pairs))



# class ScoreFn:
#     """Based on eq.3, valid for a=0"""
#     def __init__(self, score_net, params, sde, prior, marginal_prior) -> None:
#         """
#         Args:
#         score_net: trained score_network (local_score_net.py)


#         """
#         self.score_net = score_net
#         self.params = params
#         self.sde = sde
#         self.prior = prior

#         if marginal_prior is None:
#             def default_marginal_prior_score_fn(a, theta):
#                 m = sde.mu(a)
#                 std = sde.std(a)
#                 p_t_score = prior.score(theta)
#                 return p_t_score
#             self.marginal_prior = default_marginal_prior_score_fn
#         else:
#             self.marginal_prior = marginal_prior


#     def __call__(self, a, theta, x_0_T, **kwargs):

#         # compute the score of the score network (B, T-1, d_theta)
#         score = self.score_net.apply(self.params, theta, x_0_T, a)

#         # sum over all t
#         summed_local_score = jnp.sum(score, axis=1)

#         # prior score
#         prior_score = self.marginal_prior(a, theta)

#         # composition (Eq. 3)
#         N = score.shape[1]

#         global_score = (1-N) * prior_score + summed_local_score

#         if global_score.shape[0] == 1:
#             global_score = jnp.squeeze(global_score, axis=0)

#         return jnp

