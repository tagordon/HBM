import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.stats import sigma_clip
import celerite2
import os
import emcee
from multiprocess import Pool
import multiprocess.context as ctx
ctx._force_start_method('spawn')

import sys
sys.path.append('../get_planet_params/')
from distributions import *
from transit import *
from utils import *

ld_data_path = os.environ['EXO_LD_PATH']
os.environ["OMP_NUM_THREADS"] = "1"

pl_param_names = [
    't0', 'ratror', 'orbper', 'ratdor', 
    'orbincl', 'orbeccen', 'orblper'
]

def get_initial_transit_params(
    time,
    flux,
    priors, 
    st_params,
    start_wav,
    end_wav,
    disp_filt
):

    # get initial values and widths for limb darkening parameters
    u, du = get_ld_params(
        start_wav, end_wav, disp_filt, st_params
    )

    # get initial values and distribution widths from planet priors
    values = []
    widths = []
    midtimes = []
    midtime_widths = []
    for i, prior in enumerate(priors):
        d = prior.copy()
        istr = '{}'.format(i)
        keys = [k + istr for k in d.keys() if k in pl_param_names]
        v = [p.init for k, p in d.items() if k in pl_param_names]
        w = [p.width for k, p in d.items() if k in pl_param_names]
        v = [v[gk] for gk in [keys.index(k + istr) for k in pl_param_names]]
        w = [w[gk] for gk in [keys.index(k + istr) for k in pl_param_names]]
        
        midtimes.append(v.pop(0))
        midtime_widths.append(w.pop(0))
        values = values + v
        widths = widths + w
    
    return (
        np.concatenate([[u[0], u[1]], midtimes, values]), 
        np.concatenate([[du[0], du[1]], midtime_widths, widths])
    )

def build_mask(time, flux, priors, filter_window=30, out_sigma=3):
    
    out_mask = sigma_clip(
        flux - gaussian_filter1d(flux, filter_window), 
        sigma=out_sigma
    ).mask
    
    trans_mask = np.zeros_like(flux, dtype=np.bool_)
    for prior in priors:
        tmid = np.mod(prior['t0'].init - time[0], prior['orbper'].init) + time[0]
        trans_mask = trans_mask | (np.abs(time - tmid) < 1.1 * prior['trandur'].init / 48)
    
    return out_mask | trans_mask

def get_initial_params(
    time,
    flux, 
    err,
    start_wav,
    end_wav,
    disp_filt,
    detrending_vectors, 
    priors,
    st_params,
    polyorder=1,
):
        
    mask = build_mask(time, flux, priors)
    coeffs, fit = gls_fit(
        time, 
        flux, 
        detrending_vectors, 
        mask, 
        polyorder=polyorder, 
        return_coeffs=True
    )
    coeffs_widths = [1e-4] * len(coeffs)
    err_guess = np.sqrt(np.var(flux[~mask] - fit[~mask]) - np.sqrt(np.nansum(err**2)))
    p, dp = get_initial_transit_params(
        time,
        flux,
        priors, 
        st_params,
        start_wav,
        end_wav,
        disp_filt
    )

    noise_params = [err_guess]
    noise_widths = [1e-4]

    params = np.concatenate([noise_params, coeffs, p])
    widths = np.concatenate([noise_widths, coeffs_widths, dp])
        
    return params, widths, mask

def compute_priors(priors, params, u1_prior, u2_prior, ld_prior):

    n = len(priors)
    u1, u2 = params[:2]
    t0s = params[2:2+n]
    transit_params = params[2+n:]
    pr = 0
            
    for i, prior in enumerate(priors):
        pr += prior['t0'].prior(t0s[i])
        for p, name in zip(transit_params[i*6:(i+1)*6], pl_param_names[1:]):
            pr += prior[name].prior(p)

    if ld_prior:
        pr += u1_prior.prior(u1)
        pr += u2_prior.prior(u2)
    else:
        pr += uniform_prior(0, 1).prior(u1)
        pr += uniform_prior(0, 1).prior(u1)
        
    return pr

def get_model(
    p,
    time,
    detrending_vectors,
    polyorder=1
):
        
    ncoeffs = 1 + polyorder + len(detrending_vectors)
    
    err_inflate = p[0]
    coeffs = p[1:ncoeffs + 1]
    f = coeffs[0]
    p = p[1 + ncoeffs:]
        
    # number of transits based on number of provided parameters
    n = np.int64(np.floor((len(p) - 2) // 7))

    trend = get_trend_model(time, detrending_vectors, coeffs, polyorder)
    mu = 1 + keplerian_transit(time, n, p)
    
    return mu * trend # * f + trend
        

def build_logp(
    time,
    flux, 
    err,
    detrending_vectors,
    start_wav,
    end_wav,
    disp_filt,
    priors,
    st_params,
    polyorder=1, 
    ld_priors=True,
):

    n_components = len(detrending_vectors)
    ncoeffs = 1 + polyorder + n_components
        
    u1_prior, u2_prior = get_ld_priors(start_wav, end_wav, disp_filt, st_params)
    e2 = err**2

    def log_prob(p):
        
        err_inflate = p[0]
        coeffs = p[1:ncoeffs + 1]
        f = coeffs[0]
        p = p[1 + ncoeffs:]
        
        # number of transits based on number of provided parameters
        n = np.int64(np.floor((len(p) - 2) // 7))
        
        trend = get_trend_model(time, detrending_vectors, coeffs, polyorder)
        err = np.sqrt(e2 + err_inflate**2)
        
        try:

            mu = (1 + keplerian_transit(time, n, p)) * trend
            ll = log_likelihood(flux, mu, err)

            pr = compute_priors(
                priors, p, u1_prior, u2_prior, ld_priors
            )

            if np.isfinite(ll) & np.all(err > 0):
                return ll + pr
            else:
                return -np.inf
                
        except Exception as e:
            return -np.inf

    return log_prob
