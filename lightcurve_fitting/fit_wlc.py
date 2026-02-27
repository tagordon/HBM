import numpy as np
from astropy.stats import sigma_clip
from scipy.ndimage import gaussian_filter1d
from astropy.time import Time
import h5py
from wlc_utils import *

import sys
sys.path.append('../get_planet_params/')
from build_priors import get_params, get_priors
import distributions

au_per_rs = 215.0320290

def load_priors_and_parameters(target, pl_ref, st_ref):
    
    pl_author, pl_year = pl_ref
    st_author, st_year = st_ref
    
    priors = get_priors(target, author=pl_author, year=pl_year)
    params = get_params(target, author=st_author, year=st_year)
    
    if len(priors) == 0:
        raise Exception('Failed to resolve target name or planet parameters reference')
    if len(params) == 0:
        raise Exception('Failed to resolve target name or stellar parameters reference')

    st_params = {
        k[3:]:v[0] 
        for k, v in list(params.values())[0].items() 
        if ('st_' in k) | ('sy_' in k)
    }
    pl_priors = {
        k[3:]:v 
        for k, v in list(priors.values())[0].items() 
        if ('pl_' in k) & (not isinstance(v, (int, float)))
    }

    #pl_priors['tranmid'] = distributions.uniform_prior(0.0, 100.0)
    pl_priors['ratror'] = distributions.uniform_prior(0.0, 1.0, init=pl_priors['ratror'].init)
    
    if not 'ratdor' in pl_priors.keys():
        if 'orbsmax' in pl_priors.keys():
            ratdor = au_per_rs / st_params['rad'] * pl_priors['orbsmax'].init
            ratdor_err = au_per_rs / st_params['rad'] * pl_priors['orbsmax'].width
            pl_priors['ratdor'] = distributions.normal_prior(ratdor, ratdor_err)
            
    if not 'orbeccen' in pl_priors.keys():
            pl_priors['orbeccen'] = distributions.trunc_normal_prior(0, 1e-6, 0, 1, init=1e-6)
    if not 'orblper' in pl_priors.keys():
        pl_priors['orblper'] = distributions.uniform_prior(0, 180, init=90.0)
    if not 'orbincl' in pl_priors.keys():
        k = pl_priors['ratror'].init
        ecc = pl_priors['orbeccen'].init
        w = pl_priors['orblper'].init
        if 'ratdor' in pl_priors.keys():
            a = pl_priors['ratdor'].init
        else:
            raise Exception('reference has no value for semimajor axis.')
        incl_width = 180 / np.pi * np.arccos((1 + k) * rs / a * (1 + ecc * np.sin(w)) / (1 - ecc**2))
        pl_priors['orbincl'] = distributions.uniform_prior(90 - incl_width, 90 + incl_width, init=90.0)
        
    if not 'trandur' in pl_priors.keys():
        a = pl_priors['ratdor'].init
        P = pl_priors['orbper'].init
        k = pl_priors['ratror'].init
        ecc = pl_priors['orbeccen'].init
        w = pl_priors['orblper'].init * np.pi / 180
        i = pl_priors['orbincl'].init * np.pi / 180
        b = a * np.cos(i) * (1 - ecc**2) / ((1 + ecc * np.sin(w)))
        dur = P / np.pi * np.arcsin((np.sqrt((1 + k)**2 - b**2)) / (a * np.sin(i)))
        dur *= np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(w)) * 24
        pl_priors['trandur'] = distributions.constant(init=dur)
    
    return pl_priors, st_params


def load_data_jedi(d):

    time = np.load(d + '_times_bjd.npy')
    time = np.array(time, dtype=np.float64)
    time_offset = time[0]
    time -= time[0]
    spect = np.load(d + '.npy')
    wavs = np.load(d + '_wav.npy')

    xshifts = np.load(d + '_shiftx.npy')
    yshifts = np.load(d + '_shifty.npy')
    err = np.load(d + '_err.npy')

    return time, wavs, spect, err, time_offset, xshifts, yshifts

def load_data_eureka(d):

    with h5py.File(d, 'r') as file:
        time = np.array(file['time'], dtype=np.float64)
        time_offset = time[0]
        time -= time_offset
        spect = np.array(file['optspec'], dtype=np.float64)
        wavs = np.array(file['wave_1d'], dtype=np.float64)
        xshifts = np.array(file['x'], dtype=np.float64)
        yshifts = np.array(file['y'], dtype=np.float64)
        err = np.array(file['opterr'], dtype=np.float64)

    return time, wavs, spect, err, time_offset, xshifts, yshifts

def prep_data(control_dict):
    
    if control_dict['pipeline'] == 'eureka':
        load_func = load_data_eureka
    elif control_dict['pipeline'] == 'jedi':
        load_func = load_data_jedi
    else:
        load_func = load_data_eureka

    times = []
    spects = []
    wavs = []
    xshifts = []
    yshifts = []
    errs = []
    tos = []
    for d in control_dict['data_directories']:
        t, w, s, e, to, xs, ys = load_func(d)
        times.append(t)
        spects.append(s)
        wavs.append(w)
        xshifts.append(xs)
        yshifts.append(ys)
        errs.append(e)
        tos.append(to)
        
    to_jd = Time(tos[0], format='mjd', scale='tdb').jd
    
    for priors in control_dict['priors']:
        tmid = priors['tranmid'].init
        t0 = np.mod(tmid - to_jd, priors['orbper'].init)
        priors['t0'] = distributions.uniform_prior(times[0][0], times[0][-1], init=t0)

    sort = np.argsort([t[0] for t in times])
    times = [times[i] for i in sort]
    spects = [spects[i] for i in sort]
    wavs = [wavs[i] for i in sort]
    xshifts = [xshifts[i] for i in sort]
    yshifts = [yshifts[i] for i in sort]
    errs = [errs[i] for i in sort]

    mask_columns = control_dict['columns_to_mask']
    mask = np.array([True] * len(wavs[0]))
    mask[mask_columns] = False
    spects = [s[:, mask] for s in spects]
    wavs = [w[mask] for w in wavs]
    errs = [e[:, mask] for e in errs]
    fluxes = [np.nansum(s, axis=1) for s in spects]
    specs = [np.nansum(s, axis=0) for s in spects]
    
    detectors = [control_dict['detector']] * len(control_dict['data_directories'])
    
    return times, spects, errs, wavs, specs, fluxes, detectors

def get_log_probs(times, fluxes, errs, detectors, detrending_vectors, priors, st_params, polyorder=1):

    lps = []
    for t, f, e, d, dv in zip(times, fluxes, errs, detectors, detrending_vectors):
        lps.append(
            build_logp(
                t,
                f, 
                e,
                dv, 
                d,
                priors,
                st_params,
                polyorder=polyorder, 
                ld_priors=True,
            )
        )
        
    return lps

def get_joint_initial_params(
    times, 
    fluxes, 
    errs,
    detrending_vectors,
    detectors, 
    priors, 
    st_params, 
    polyorder,
):
    
    other_params = []
    other_widths = []
    for i, (t, f, e, d, dv) in enumerate(
        zip(times, fluxes, errs, detectors, detrending_vectors)
    ):
        initial_params, widths = get_initial_params(
            t,
            f, 
            e,
            d, 
            dv,
            priors,
            st_params,
            polyorder=polyorder,
        )

        n_other = polyorder + 5 + len(detrending_vectors[0])
            
        other_params.append(initial_params[:n_other])
        other_widths.append(widths[:n_other])
        if i == 0:
            transit_params = initial_params[n_other:]
            transit_widths = widths[n_other:]

    return (
        np.concatenate([np.concatenate(other_params), transit_params]),
        np.concatenate([np.concatenate(other_widths), transit_widths])
    )

def get_joint_log_prob(lps, n_components, polyorder=1):

    n_lcs = len(lps)
    n_op = polyorder + 5 + n_components        
    n_other_params = n_op * n_lcs
    
    def log_prob(p):

        transit_params, other_params = p[n_other_params:], p[:n_other_params]
        
        total_prob = 0
        for i in range(n_lcs):
            op = other_params[:n_op]
            other_params = other_params[n_op:]
            total_prob += lps[i](np.concatenate([op, transit_params]))
        return total_prob

    return log_prob

def run_mcmc(
    times,
    fluxes, 
    errs,
    detrending_vectors,
    detectors, 
    priors,
    st_params,
    polyorder=1, 
    samples=10_000,
    progress=True,
    nproc=1,
):
    
    params, widths = get_joint_initial_params(
        times, 
        fluxes, 
        errs,
        detrending_vectors,
        detectors, 
        priors, 
        st_params,  
        polyorder,
    )
        
    lps = get_log_probs(
        times, 
        fluxes, 
        errs,
        detectors, 
        detrending_vectors, 
        priors, 
        st_params, 
        polyorder=polyorder,
    )
    log_prob = get_joint_log_prob(
        lps, 
        len(detrending_vectors), 
        polyorder, 
    )

    widths = np.array([np.min([w, 1e-4]) for w in widths])
    pos = params + widths * np.random.randn(len(params)*2, len(params))
    nwalkers, ndim = pos.shape

    if nproc == 1:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob
        )
        sampler.run_mcmc(
            pos, 
            samples, 
            progress=progress, 
            skip_initial_state_check=True
        );
        
    elif nproc > 1:
        
        with Pool(nproc) as pool:
            
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_prob, pool=pool
            )
            sampler.run_mcmc(
                pos, 
                samples, 
                progress=progress, 
                skip_initial_state_check=True
            );
            
    return sampler

def set_optional_params(control_dict):
    
    optional_params = ['polyorder', 'detrending_vectors', 'out_filter_width', 'out_sigma', 'progress']
    defaults = [1, [[]], 50, 4, True]
    
    for k, d in zip(optional_params, defaults):
        
        try:
            control_dict[k]
        except:
            control_dict[k] = d
    
    return control_dict

def fit(control_dict):
    
    times, spects, errs, wavs, _, _, detectors = prep_data(control_dict)
    
    control_dict = set_optional_params(control_dict)
    detrending_vectors = np.array(control_dict['detrending_vectors'] * len(times))
    
    n = len(times)
    fluxes = [np.nansum(s, axis=1) for s in spects]
    flux_err = [np.sqrt(np.nansum(e**2, axis=1)) for e in errs]
        
    n_sys_params_per_transit = control_dict['polyorder'] + len(control_dict['detrending_vectors'][0]) + 5
    n_sys_params = n_sys_params_per_transit * n
      
    # first transit should start at t=0 
    #to = times[0][0]
    #for i in range(len(times)):
    #    times[i] -= to

    # build outlier masks 
    masks = []
    for flux in fluxes:
        masks.append(
            sigma_clip(
                flux - gaussian_filter1d(flux, control_dict['out_filter_width']), 
                sigma=control_dict['out_sigma']
            ).mask
        )

    sampler = run_mcmc(
        [t[~m] for t, m in zip(times, masks)],
        [f[~m] for f, m in zip(fluxes, masks)], 
        [e[~m] for e, m in zip(flux_err, masks)],
        [dv[~m] if len(dv)==len(m) else [] for dv, m in zip(detrending_vectors, masks)],
        detectors, 
        control_dict['priors'],
        control_dict['stellar_parameters'], 
        polyorder=control_dict['polyorder'], 
        samples=control_dict['samples'],
        progress=control_dict['progress'],
        nproc=control_dict['num_proc'],
    )

    chains = sampler.get_chain()[control_dict['burnin']:, :, :]

    results = []
    transit_params, other_params = chains[:, :, n_sys_params:], chains[:, :, :n_sys_params]
    
    for i in range(n):

        op = other_params[:, :, :n_sys_params_per_transit]
        other_params = other_params[:, :, n_sys_params_per_transit:]
        chain = np.concatenate([op, transit_params], axis=2)
            
        transit_param_names = ['tranmid', 'ratror', 'orbper', 'ratdor', 
            'orbincl', 'orbeccen', 'orblper'
        ]
        transit_param_labels = ['u1', 'u2']
        for j, prior in enumerate(control_dict['priors']):
            transit_param_labels = transit_param_labels + [n + '{}'.format(j) for n in transit_param_names]
            
        results.append(
            {
                'control_dict': control_dict,
                'detrending_vectors': detrending_vectors[i],
                'polyorder': control_dict['polyorder'],
                'detector': detectors[i],
                'chain': chain,
                'mask': masks[i],
                'time': times[i],
                'spect': spects[i],
                'wavs': wavs[i],
                'flux': fluxes[i],
                'flux_err': flux_err[i],
                'errs': errs[i],
                'labels': (
                    ['err_factor'] 
                    + ['p{0}'.format(i) for i in range(control_dict['polyorder'] + 1)] 
                    + ['c{0}'.format(i) for i in range(len(detrending_vectors[i]))] 
                    + transit_param_labels
                )
            }
        )
        
        for r in results:
            r['posterior'] = {k: v for k, v in zip(r['labels'], r['chain'].T)}

    return results

def get_model_samples(result, n=None):
    
    if n is None:
        
        p = np.median(result['chain'], axis=(0, 1))
        return get_model(
            p,
            result['time'],
            detrending_vectors=result['detrending_vectors'],
            polyorder=result['polyorder']
        )
    
    else:
        samples = np.concatenate(result['chain'], axis=0)
        inds = np.random.randint(samples.shape[0], size=n)
        get_single_model = lambda p: get_model(
            p, 
            result['time'], 
            detrending_vectors=result['detrending_vectors'], 
            polyorder=result['polyorder']
        )
        return [get_single_model(p) for p in samples[inds]]
    
def check_initial_state(control_dict):
    
    times, spects, errs, wavs, _, fluxes, detectors = prep_data(control_dict)
    
    try:
        detrending_vectors = control_dict['detrending_vectors']
    except:
        detrending_vectors = np.array([[]] * len(times))
    
    models = []
    for time, spect, err, flux, dv, detector in zip(times, spects, errs, fluxes, detrending_vectors, detectors):

        inits, _ = get_initial_params(
            time,
            flux, 
            err,
            detector, 
            dv,
            control_dict['priors'],
            control_dict['stellar_parameters'],
            polyorder=control_dict['polyorder'],
        )
        
        models.append(
            get_model(
                inits,
                time,
                dv,
                polyorder=control_dict['polyorder']
            )
        )
        
    return models