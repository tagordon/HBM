import numpy as np
import batman
from distributions import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def keplerian_transit(t, n, p):
    
    u1, u2 = p[:2]
    params = p[2:]
    
    m = np.zeros_like(t)
    for i in range(n):
        
        t0, r, p, a, i, e, w = params[i*7:(i+1)*7]
        param_obj = batman.TransitParams()
        param_obj.t0 = t0                     #time of inferior conjunction
        param_obj.per = p                     #orbital period
        param_obj.rp = r                      #planet radius (in units of stellar radii)
        param_obj.a = a                       #semi-major axis (in units of stellar radii)
        param_obj.inc = i                     #orbital inclination (in degrees)
        param_obj.ecc = e                     #eccentricity
        param_obj.w = w                       #longitude of periastron (in degrees)
        param_obj.u = [u1, u2]                #limb darkening coefficients [u1, u2]
        param_obj.limb_dark = "quadratic"     #limb darkening model

        m += batman.TransitModel(param_obj, t, fac=0.01).light_curve(param_obj) 
    
    return m - n