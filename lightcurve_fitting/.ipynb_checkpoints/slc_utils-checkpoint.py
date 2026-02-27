import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip
import emcee
from multiprocess import Pool
import multiprocess.context as ctx
ctx._force_start_method('spawn')

from distributions import *
from transit import *
from utils import *

def build_mask(time, flux, transit_params, mask_buffer=50):
    
    mask = sigma_clip(flux - gaussian_filter1d(flux, 50), sigma=3).mask
    
    for params in transit_params:
        t0, r, per, a, inc, e, w = params

        b = a * np.cos(inc) * (1 - e**2) / (1 + e * np.sin(w))
        C = np.sqrt((1 + r)**2 - b**2)

        dur = per / np.pi * np.arcsin(
            C / (a * np.sin(inc))
        ) * np.sqrt(1 - e**2) / (1 + e * np.sin(w))

        start_mask = np.argmin(np.abs(time - (t0 - 0.5 * dur))) - mask_buffer
        end_mask = np.argmin(np.abs(time - (t0 + 0.5 * dur))) + mask_buffer

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
    # t0, r... t0, r...
    fixed_params = fixed_params[2:]
    n = len(fixed_params) // 7
    transit_params = [fixed_params[i*7:(i+1)*7] for i in range(n)]
    #u1, u2, t0, r_wlc, per, a, inc, e, w = fixed_params
    mask = build_mask(time, flux, transit_params)
    coeffs, fit = gls_fit(
        time, 
        flux, 
        [],
        mask=mask, 
        polyorder=polyorder, 
        return_coeffs=True
    )
    
    err_guess = np.std(flux[~mask] - fit[~mask])
    rads = [tp[1] for tp in transit_params]
    # r1, r2, u1, u2, err, p1, p2
    params = np.concatenate([rads, [u1, u2, err_guess], coeffs])

    return params

def build_logp(
    time,
    flux, 
    fixed_params,
    stellar_params,
    start_wav,
    end_wav,
    polyorder=1, 
):

    ncoeffs = 1 + polyorder

    u1, u2 = fixed_params[:2]
    fixed_params = fixed_params[2:]
    n = len(fixed_params) // 7
    #transit_params = [fixed_params[i*7:(i+1)*7] for i in range(n)]
    
    u1_prior, u2_prior = get_ld_priors(start_wav, end_wav, stellar_params)
    #t0, r_wlc, per, a, inc, e, w = fixed_params

    def log_prob(p):

        # p = r1, r2, u1, u2, err, p1, p2
        
        rads = p[:n]
        u1, u2 = p[n:n + 2]
        err = p[n + 2]
        coeffs = p[-ncoeffs:]
        f = coeffs[0]
        
        #print('\nrads: {}'.format(rads), '\nu: {}'.format([u1, u2]), '\nerr: {}'.format(err), '\ncoeffs: {}'.format(coeffs))
        trend = get_trend_model(time, [], coeffs[1:], polyorder)

        for i in range(n):
            fixed_params[i * 7 + 1] = rads[i]
            
        #try:

        #p = [u1, u2, t0, r, per, a, inc, e, w]
        mu = 1 + keplerian_transit(time, n, np.concatenate([[u1, u2], fixed_params]))
        mu = mu * f + trend
        
        ll = log_likelihood(flux, mu, err)

        pr = u1_prior.prior(u1)
        pr += u2_prior.prior(u2)
        pr += np.sum([uniform_prior(0, 1).prior(r) for r in rads])
        
        if np.isfinite(ll) & (err > 0):
            return ll + pr
        else:
            return -np.inf

        #except Exception as e:
        #    return -np.inf

    return log_prob