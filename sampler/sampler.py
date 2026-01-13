import jax
import jax.numpy as jnp
from jax.scipy.special import gamma

jax.config.update("jax_enable_x64", True)

gauss_like = lambda f, y, phi: jnp.exp(-0.5 * (f - y)**2 / phi[0]**2) / jnp.sqrt(2 * jnp.pi * phi[0]**2)

def t_like(f, y, phi): 

    nu, tau = phi
    norm_fac = gamma((nu + 1) * 0.5) / (gamma(nu * 0.5) * tau * jnp.sqrt(nu * jnp.pi))
    alpha = -0.5 * (nu + 1)
    return norm_fac * (1 + (f - y)**2 / (nu * tau**2))**alpha

t_like_df1 = lambda f, y, phi: t_like(f, y, [phi[0], 1])
t_like_df3 = lambda f, y, phi: t_like(f, y, [phi[0], 3])
t_like_df5 = lambda f, y, phi: t_like(f, y, [phi[0], 5])

def HBM_log_likelihood(x, theta, phi, samples, trend_func, like_func, norm_func=lambda phi: 0, seed=0, k=1000):

    kj = jax.random.randint(jax.random.key(seed), shape=k, minval=0, maxval=samples.shape[1])
    Pns = like_func(trend_func(x, theta)[:, None], samples[:, kj], phi)
    ll = jnp.sum(jnp.log(jnp.sum(Pns, axis=1)), axis=0) + norm_func(phi)
    return jax.lax.cond(jnp.isfinite(ll), lambda: ll, lambda: -jnp.inf)