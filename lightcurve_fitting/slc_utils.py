import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip
import emcee
from multiprocess import Pool
import multiprocess.context as ctx
ctx._force_start_method('spawn')

from transit import *
from utils import *

import matplotlib.pyplot as plt

def build_mask(time, flux, transit_params, t0s, mask_buffer=50):
    
    mask = sigma_clip(flux - gaussian_filter1d(flux, 50), sigma=3).mask
    
    for t0, params in zip(t0s, transit_params):
        r, per, a, inc, e, w = params
                
        tmid = np.mod(t0 - time[0], per) + time[0]

        b = a * np.cos(inc * np.pi / 180) * (1 - e**2) / (1 + e * np.sin(w * np.pi / 180))
        C = np.sqrt((1 + r)**2 - b**2)

        dur = per / np.pi * np.arcsin(
            C / (a * np.sin(inc * np.pi / 180))
        ) * np.sqrt(1 - e**2) / (1 + e * np.sin(w * np.pi / 180))
                
        start_mask = np.argmin(np.abs(time - (tmid - 0.5 * dur))) - mask_buffer
        end_mask = np.argmin(np.abs(time - (tmid + 0.5 * dur))) + mask_buffer

        trans_mask = np.zeros_like(flux, dtype=np.bool_)
        trans_mask[start_mask:end_mask] = True
        mask = mask | trans_mask
         
    return mask

def get_initial_params(
    time,
    flux, 
    fixed_params,
    polyorder=1,
):

    u1, u2 = fixed_params[:2]
    n = (len(fixed_params) - 2) // 7
    t0s = fixed_params[2:2+n]
    fixed_params = fixed_params[2+n:]
    transit_params = [fixed_params[i*6:(i+1)*6] for i in range(n)]    
    mask = build_mask(time, flux, transit_params, t0s)
    
    coeffs, fit = gls_fit(
        time, 
        flux, 
        [],
        mask=mask, 
        polyorder=polyorder, 
        return_coeffs=True
    )
    
    err_guess = np.std(flux[~mask] - fit[~mask])
    rads = [tp[0] for tp in transit_params]
    params = np.concatenate([rads, [u1, u2, err_guess], coeffs])

    return params

def build_logp(
    time,
    flux, 
    fixed_params,
    stellar_params,
    start_wav,
    end_wav,
    disp_filt,
    polyorder=1, 
):

    ncoeffs = 1 + polyorder

    # t1, t2... | r, p, a, i, e, w | r, p, a, i, e, w | ... 
    u1, u2 = fixed_params[:2]
    n = (len(fixed_params) - 2) // 7
    t0s = fixed_params[2:2+n]
    fixed_params = fixed_params[2+n:]
        
    u1_prior, u2_prior = get_ld_priors(start_wav, end_wav, disp_filt, stellar_params)

    def log_prob(p):
        
        rads = p[:n]
        u1, u2 = p[n:n + 2]
        err = p[n + 2]
        coeffs = p[-ncoeffs:]
        f = coeffs[0]
                
        trend = get_trend_model(time, [], coeffs, polyorder)

        for i in range(n):
            fixed_params[i * 6] = rads[i]
        
        try:

            mu = (1 + keplerian_transit(time, n, np.concatenate([[u1, u2], t0s, fixed_params]))) * trend
        
            ll = log_likelihood(flux, mu, err)

            pr = u1_prior.prior(u1)
            pr += u2_prior.prior(u2)
        
            if np.isfinite(ll) & (err > 0):
                return ll + pr
            else:
                return -np.inf

        except Exception as e:
            return -np.inf

    return log_prob
