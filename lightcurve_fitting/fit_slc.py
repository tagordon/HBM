import os
import numpy as np
from slc_utils import *
import utils
from scipy.stats import gaussian_kde

def get_log_probs(
    times, 
    fluxes, 
    fixed_params, 
    start_wav, 
    end_wav, 
    disp_filt,
    st_params_dict, 
    polyorder=1,
):

    lps = []
    for t, f, fp in zip(times, fluxes, fixed_params):
        
        lps.append(
            build_logp(
                t,
                f, 
                fp,
                st_params_dict,
                start_wav,
                end_wav,
                disp_filt,
                polyorder=polyorder, 
            )
        )
        
    return lps

def get_joint_initial_params(
    times, 
    fluxes, 
    fixed_params, 
    polyorder=1,
):

    nplanets = (len(fixed_params[0]) - 2) // 7
    other_params = []
    for i, (t, f, fp) in enumerate(
        zip(times, fluxes, fixed_params)
    ):
        # r1, r2, u1, u2, err, p1, p2
        initial_params = get_initial_params(
            t,
            f, 
            fp, 
            polyorder=polyorder,
        )
            
        other_params.append(initial_params[2 + nplanets:])
        if i == 0:
            transit_params = initial_params[:2 + nplanets]

    return np.concatenate([transit_params, np.concatenate(other_params)])
    #return np.concatenate([np.concatenate(other_params), transit_params])

# problem in here somewhere.
def get_joint_log_prob(lps, polyorder):

    n_lcs = len(lps)
    n_other_params = (polyorder + 2) * n_lcs

    def log_prob(p):

        # r1, r2, u1, u2, err, p1, p2
        transit_params, other_params = p[:-n_other_params], p[-n_other_params:]
        #print(p, n_other_params)

        total_prob = 0
        for i in range(len(lps)):

            n_op = polyorder + 2
            op = other_params[:n_op]
            other_params = other_params[n_op:]
            total_prob += lps[i](np.concatenate([transit_params, op]))
            
        return total_prob

    return log_prob

def run(
    times,
    specs, 
    wl_params,
    stellar_params,
    wav_bin_edges,
    masks,
    disp_filt,
    polyorder, 
    progress,
    nproc,
    samples,
):
       
    ncoeffs = 1 + polyorder
    
    # u1, u2, t1, t2, transit_params1, transit_params2
    fixed_params = [wlp[1 + ncoeffs:] for wlp in wl_params]
    masks = np.array([np.tile(masks[i], (1, specs[i].shape[1])).T for i in range(len(masks))])

    def run_single_band(ind):

        mask = np.array(~masks[:, ind])

        # r1, r2, u1, u2, err, p1, p2... 
        params = get_joint_initial_params(
            [t[m] for t, m in zip(times, mask)], 
            [s[m, ind] for s, m in zip(specs, mask)], 
            fixed_params, 
            polyorder=polyorder,
        )

        lps = get_log_probs(
            [t[m] for t, m in zip(times, mask)], 
            [s[m, ind] for s, m in zip(specs, mask)], 
            fixed_params, 
            wav_bin_edges[ind],
            wav_bin_edges[ind + 1], 
            disp_filt,
            stellar_params, 
            polyorder=polyorder,
        )
        
        log_prob = get_joint_log_prob(
            lps, 
            polyorder, 
        )
        
        pos = params + 1e-4 * np.random.randn(len(params)*2, len(params))
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob
        )
        
        sampler.run_mcmc(
            pos, 
            samples, 
            progress=False, 
            skip_initial_state_check=True
        );
            
        return sampler

    posterior = []
    means = []
    stds = []

    if nproc == 1:

        for i in range(specs[0].shape[1]):

            sampler = run_single_band(i)
            posterior.append(sampler)

    else:

        nbands = specs[0].shape[1]
        nbatches = nbands // nproc
        remainder = np.remainder(nbands, nproc)
        
        if remainder > 0:
            nbatches += 1
        
        band_inds = np.arange(nbands)
        batches = [band_inds[i * nproc:(i + 1) * nproc] for i in range(nbatches)]

        for b in batches:

            if progress:
                print('\r                                                            ', end='')
                print('\rrunning bands {}-{} of {}'.format(b[0] + 1, b[-1] + 1, nbands), end='')
            with Pool(nproc) as pool:

                samplers_or_means = pool.map(run_single_band, b)

            posterior.append(samplers_or_means)

        posterior = np.concatenate(posterior)

    return posterior

def set_optional_params(control_dict):
    
    optional_params = [
        'wav_per_bin', 'pix_per_bin', 
        'slc_polyorder', 'out_filter_width', 
        'out_sigma', 'start_wav', 'end_wav',
        'num_proc_slc'
    ]
    defaults = [0.02, None, control_dict['polyorder'], 50, 4, 2.87, 5.17, 1]
    
    for k, d in zip(optional_params, defaults):
        
        try:
            control_dict[k]
        except:
            control_dict[k] = d
    
    return control_dict

def crop(specs, errs, wavs, start_wav, end_wav):
    
    specs = [s[:, (wavs > start_wav) & (wavs < end_wav)] for s in specs]
    errs = [e[:, (wavs > start_wav) & (wavs < end_wav)] for e in errs]
    wavs = wavs[(wavs > start_wav) & (wavs < end_wav)]
        
    return specs, errs, wavs

def fit(wlc_result, samples=None):

    if samples is None:
        samples = wlc_result[0]['control_dict']['samples_slc']
    
    ntrans = len(wlc_result)
    control_dict = set_optional_params(wlc_result[0]['control_dict'])
    nplanets = len(wlc_result[0]['control_dict']['priors'])

    polyorder = control_dict['slc_polyorder']

    times = [res['time'] for res in wlc_result]
    specs = [res['spect'] for res in wlc_result]
    errs = [res['errs'] for res in wlc_result]
    stellar_params = control_dict['stellar_parameters']
    #start_wav = wlc_result[0]['start_wav']
    #end_wav = wlc_result[0]['end_wav']
    wavs = wlc_result[0]['wavs']
            
    wl_params = []
    for result in wlc_result:

        chain = result['chain']
        ncoeffs = 1 + control_dict['polyorder']

        op, p = chain[:, :, :2 + ncoeffs], chain[:, :, 2 + ncoeffs:]
                        
        p = np.concatenate(p, axis=0).T
        op = np.concatenate(op, axis=0).T
                
        p = np.vstack([op, p])
        isfin = np.all(np.isfinite(p), axis=0)
        p = p[:, isfin]
    
        wl_vals = []
        for s in p:
            x = np.linspace(s.min(), s.max(), 100)
            y = gaussian_kde(s.flatten()).evaluate(x)
            wl_vals.append(x[np.argmax(y)])
            
        wl_params.append(wl_vals)
         
    times = [np.array(t, dtype=np.float64) for t in times]
    specs, errs, wavs = crop(specs, errs, wavs, control_dict['start_wav'], control_dict['end_wav'])

    wav_per_bin, pix_per_bin = control_dict['wav_per_bin'], control_dict['pix_per_bin']
    if (wav_per_bin is None) & (pix_per_bin is None):
        binned_wavs = wavs
        binned_specs = specs
        binned_errs = errs
    elif wav_per_bin is None:
        binned_wavs = np.array(
            [np.sum(
                wavs[pix_per_bin * i: pix_per_bin * (i + 1)]
            ) for i in range(np.int64(len(wavs) // pix_per_bin))]
        )
        wav_bin_edges = np.array(
            [wavs[pix_per_bin * i] for i in range(np.int64(len(wavs) // pix_per_bin + 1))]
        )

        binned_specs = []
        binned_errs = []
        for spec, err in zip(specs, errs):
            binned_specs.append(
                np.array(
                    [np.nansum(
                        spec[:, pix_per_bin * i: pix_per_bin * (i + 1)], axis=1
                    ) for i in range(len(wavs) // pix_per_bin)]
                ).T
            )
            binned_errs.append(
                np.array(
                    [np.sqrt(np.nansum(
                        err[:, pix_per_bin * i: pix_per_bin * (i + 1)]**2, axis=1
                    )) for i in range(len(wavs) // pix_per_bin)]
                ).T
            )
    else:
        nbands = np.int64((wavs[-1] - wavs[0]) // wav_per_bin)
        wav_bin_edges = np.linspace(wavs[0], wavs[-1], nbands + 1)
        binned_wavs = wav_bin_edges[:-1] + 0.5 * np.diff(wav_bin_edges)
    
        binned_specs = []
        binned_errs = []
        for spec, err in zip(specs, errs):
            binned_specs.append(
                np.array([
                    np.nansum(
                        spec[:, np.where(
                            (wavs >= wav_bin_edges[i]) & 
                            (wavs <= wav_bin_edges[i+1])
                        )[0]],
                        axis=1
                    )
                    for i in range(nbands)
                ]).T
            )
            binned_errs.append(
                np.array([
                    np.sqrt(np.nansum(
                        err[:, np.where(
                            (wavs >= wav_bin_edges[i]) & 
                            (wavs <= wav_bin_edges[i+1])
                        )[0]]**2,
                        axis=1
                    ))
                    for i in range(nbands)
                ]).T
            )
            
    filts = [gaussian_filter1d(bs, control_dict['out_filter_width'], axis=0) for bs in binned_specs]
    masks = [sigma_clip(bs - f, sigma=control_dict['out_sigma']).mask for bs, f in zip(binned_specs, filts)]

    post = run(
        times,
        binned_specs, 
        wl_params,
        stellar_params,
        wav_bin_edges,
        masks,
        control_dict['disp_filt'],
        polyorder,
        control_dict['progress'],
        control_dict['num_proc_slc'],
        samples
    )

    # r1, r2, u1, u2, err, p1, p2
    param_names = []
    param_names.append(np.concatenate([['r{}'.format(i) for i in range(nplanets)], ['u1', 'u2']]))
    for i in range(ntrans):
        noise_params = ['error[{0}]'.format(i)]
        poly_params = ['p{1}[{0}]'.format(i, j) for j in range(polyorder + 1)]
        other_params = np.concatenate([noise_params, poly_params])
        param_names.append(
            other_params
        )
    param_names = np.concatenate(param_names)
        

    result_dir = {
        'control_dict': control_dict,
        'nplanets': nplanets,
        'wl_params': wl_params,
        'wavs': binned_wavs,
        'lightcurves': binned_specs,
        'posteriors': [
            {
                k: v.T for k, v in zip(param_names, chain.T)
            } for chain in [p.get_chain() for p in post]
        ],
        'param_names': param_names,
        'chains': [p.get_chain() for p in post]
    }

    return result_dir
