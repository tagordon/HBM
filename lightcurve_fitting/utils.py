import numpy as np
import os
import exotic_ld
from exotic_ld import StellarLimbDarkening
import sys
sys.path.append('/data/tagordon/subneps/src/get_planet_params/')
from distributions import *

ld_data_path = os.environ['EXO_LD_PATH']

log_likelihood = lambda y, mu, err: -0.5 * np.sum((y - mu)**2 / (err**2) + np.log(err**2))

disp_filt_options = [
    'PRISM/CLEAR',
    'G395H/F290LP_NRS1',
    'G395H/F290LP_NRS2',
    'G235H/F170LP_NRS1',
    'G235H/F170LP_NRS2',
    'G140H/F100LP_NRS1',
    'G140H/F100LP_NRS2',
    'G140H/F070LP',
    'G395M/F290LP',
    'G235M/F170LP',
    'G140M/F100LP',
    'G140M/F070LP'
]

wav_ranges = [
    (0.6, 5.3),
    (2.87, 3.69),
    (3.79, 5.14),
    (1.66, 2.18),
    (2.24, 3.05),
    (0.97, 1.3),
    (1.34, 1.82),
    (0.81, 1.27),
    (2.87, 5.10),
    (1.66, 3.07),
    (0.97, 1.84),
    (0.70, 1.27)
]

# ‘JWST_NIRSpec_Prism’, ‘JWST_NIRSpec_G395H’, ‘JWST_NIRSpec_G395M’,
#‘JWST_NIRSpec_G235H’, ‘JWST_NIRSpec_G235M’, ‘JWST_NIRSpec_G140H’,
#‘JWST_NIRSpec_G140M-f100’, ‘JWST_NIRSpec_G140H-f070’, ‘JWST_NIRSpec_G140M-f070’

exotic_ld_modes = [
    'JWST_NIRSpec_Prism',
    'JWST_NIRSpec_G395H',
    'JWST_NIRSpec_G395H',
    'JWST_NIRSpec_G235H',
    'JWST_NIRSpec_G235H',
    'JWST_NIRSpec_G140H',
    'JWST_NIRSpec_G140H',
    'JWST_NIRSpec_G140H-f070',
    'JWST_NIRSpec_G395M',
    'JWST_NIRSpec_G235M',
    'JWST_NIRSpec_G140M-f100',
    'JWST_NIRSpec_G140M-f070'
]

def get_wav_ranges(disp_filt):
    
    try:
        start_wav, end_wav = wav_ranges[disp_filt_options.index(disp_filt)]
    except:
        raise Exception('invalid disperser/filter combination')
    
    return start_wav, end_wav

def get_ld_params(start_wav, end_wav, disp_filt, stellar_params):

    try:
        met = stellar_params['met']
    except:
        print('Caution: No stellar metallicity information. Defaulting to [Fe/H]=0.0')
        met = 0.0

    sld = StellarLimbDarkening(
        M_H=met, 
        Teff=stellar_params['teff'], 
        logg=stellar_params['logg'],
        ld_model="mps1",
        ld_data_path=ld_data_path
    )
    
    u, du = sld.compute_quadratic_ld_coeffs(
        wavelength_range=[start_wav * 1e4, end_wav * 1e4],
        mode=exotic_ld_modes[disp_filt_options.index(disp_filt)],
        return_sigmas=True
    )
    return u, du

def get_ld_priors(start_wav, end_wav, disp_filt, stellar_params):

    u, du = get_ld_params(start_wav, end_wav, disp_filt, stellar_params)
    u1_prior = trunc_normal_prior(u[0], du[0], 0, 1)
    u2_prior = trunc_normal_prior(u[1], du[1], 0, 1)
    return u1_prior, u2_prior

def gls_fit(time, flux, vectors, mask, polyorder=1, return_coeffs=False):
    
    time_terms_masked = np.array(
        [time[~mask]**i for i in np.arange(polyorder + 1)]
    )
    time_terms = np.array(
        [time**i for i in np.arange(polyorder + 1)]
    )

    if len(vectors) == 0:
        
        P = np.concatenate([
            time_terms_masked.T,
        ], axis=1)

        coeffs = np.linalg.inv(P.T @ P) @ (P.T @ flux[~mask])
    
        P = np.concatenate([
            time_terms.T,
        ], axis=1)

        if return_coeffs:
            return coeffs, P @ coeffs
        else:
            return P @ coeffs
    else:
    
        P = np.concatenate([
            time_terms_masked.T,
            vectors[~mask, :],
        ], axis=1)

        coeffs = np.linalg.inv(P.T @ P) @ (P.T @ flux[~mask])

        P = np.concatenate([
            time_terms.T,
            vectors,
        ], axis=1)

        if return_coeffs:
            return coeffs, P @ coeffs
        else:
            return P @ coeffs

def get_trend_model(time, vectors, coeffs, polyorder):

    time_terms = np.array(
        [time**i for i in np.arange(polyorder + 1)]
    )

    if len(vectors)==0:
        P = np.concatenate([
            time_terms.T,
        ], axis=1)
    else:
        P = np.concatenate([
            time_terms.T,
            vectors,
        ], axis=1)
        
    return P @ coeffs
