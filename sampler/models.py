import jax
import jax.numpy as jnp
import interpax

f_quad = jnp.vectorize(lambda x, theta: theta[0] + x * theta[1] + x**2 * theta[2], excluded=(1, 2))
f_lin = jnp.vectorize(lambda x, theta: theta[0] + x * theta[1], excluded=(1, 2))

def f_lin_2d(xy, theta): 

    a, bx, by = theta
    x, y = xy.T
    return a + bx * x + by * y
    
def f_quad_2d(xy, theta): 

    a, bx, by, cx, cy = theta
    x, y = xy.T
    return a + bx * x + by * y + cx * x**2 + cy * y**2

def spline_model(xy, theta):

    x, y = xy.T
    nx, ny = theta.shape
    xx = jnp.linspace(jnp.min(x), jnp.max(x), nx)
    yy = jnp.linspace(jnp.min(y), jnp.max(y), ny)
    
    return interpax.interp2d(x, y, xx, yy, theta.reshape(nx, ny))