import os
os.environ['EXO_LD_PATH'] = '/data/tagordon/exotic_ld_data'
os.environ['OMP_NUM_THREADS'] = '1'
import pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

import sys
sys.path.append('/data/tagordon/subneps/src/get_planet_params/')
from fit_wlc import *
from fit_wlc import get_model_samples
import distributions
import argparse
import glob
import copy

def main():

    cli = argparse.ArgumentParser()

    # name(s) of planet(s) -- does not need to match 
    # the exoplanet archive name, but must be a valid 
    # alias of the planet name or code will fail to 
    # download the planet parameters for building priors 
    cli.add_argument(
        '--planets',
        type = str,
        nargs = '*',
        default = []
    )

    # directorie(s) to search for .h5 file outputs 
    # of Eureka stage 3
    cli.add_argument(
        '--data_dirs',
        type = str,
        nargs = '*',
        default = [],
    )

    # directory to write outputs 
    cli.add_argument(
        '--out_dir',
        type = str,
        default = []
    )

    # references for planet parameters 
    # in the form author_yyyy where yyyy 
    # is the four-digit year of the reference.
    cli.add_argument(
        '--planet_refs',
        type = str,
        nargs = '*',
        default = [],
    )

    # references for stellar parameters in the 
    # form author_yyyy where yyy is the four-digit 
    # year of the reference. 
    cli.add_argument(
        '--stellar_ref',
        type = str,
        default = '',
    )

    # number of processors to use. Recommend --nproc 2
    cli.add_argument(
        '--nproc',
        type = np.int64,
        default = 1
    )

    # optional offset in bjd between the expected time of 
    # transit as determined by the planet parameters reference 
    # tmid time and the actual time of the transit in the 
    # data. May be necessary for planets with TTVs or 
    # stale ephemerides. 
    cli.add_argument(
        '--delta_t0',
        type = np.float64,
        default = 0.0,
    )
    
    # disperser/filter_(detector) combination for the 
    # observation. Valid options are: 
    # 'PRISM/CLEAR',
    # 'G395H/F290LP_NRS1',
    # 'G395H/F290LP_NRS2',
    # 'G235H/F170LP_NRS1',
    # 'G235H/F170LP_NRS2',
    # 'G140H/F100LP_NRS1',
    # 'G140H/F100LP_NRS2',
    # 'G140H/F070LP',
    # 'G395M/F290LP',
    # 'G235M/F170LP',
    # 'G140M/F100LP',
    # 'G140M/F070LP'
    cli.add_argument(
        '--disp_filt',
        type = str,
        default = ''
    )

    args = cli.parse_args()

    # if no outdir, place output in current directory 
    if args.out_dir == '':
        args.out_dir = args.data_dirs[0]
    if args.stellar_ref == '':
        args.stellar_ref = args.planet_refs[0]
        
    # check that there's a reference for each planet 
    if len(args.planets) != len(args.planet_refs):
        raise Exception('number of planetary parameter references does not equal number of planets.')
        
    # check that disperser/filter combination is valid, and get wavelength range
    if args.disp_filt == '':
        raise Exception('must specifiy disperser/filter combination')
    else:
        start_wav, end_wav = get_wav_ranges(args.disp_filt)
    
    # download parameters for each planet + star and build a set of priors for the 
    # planet parameters 
    st_name = '_'.join(args.stellar_ref.split('_')[:-1])
    st_year = args.stellar_ref.split('_')[-1]
    priors = []
    for planet, ref in zip(args.planets, args.planet_refs):

        pl_name = '_'.join(ref.split('_')[:-1])
        pl_year = ref.split('_')[-1]

        pl, st_params = load_priors(planet.replace('_', ' '), (pl_name, pl_year), (st_name, st_year))
        priors.append(pl)

    # find the .h5 files corresponding to the output of Eureka stage 3. 
    # Caution: This could go wrong if there are extraneous .h5 files in 
    # the directory. Note also that the directory can contain outputs 
    # from multiple visits, but should only contain outputs for a 
    # single filter/disperser_(detector) and for a single target.
    files = []
    for d in args.data_dirs:
        files.append(glob.glob(d + '/*.h5', recursive=True))
    files = np.concatenate(files)
    
    # may want to print out the files for debugging purposes 
    #print(files, d + '/*.h5')

    # options for lightcurve fitting stage. 
    control_dict = {
        'pipeline': 'eureka', # only valid option for this stage4.py file, but could in principle use JEDI outputs as well 
        'planets': args.planets, 
        'priors': priors, 
        'stellar_parameters': st_params,
        'data_directories': files,
        'columns_to_mask': [], # place holder for possibility of allowing custom column masking as a command line input later 
        'delta_t0': args.delta_t0,
        'disp_filt': args.disp_filt,
        'start_wav': start_wav,
        'end_wav': end_wav,
        'samples': 100_000, # total number of MCMC samples per chain
        'burnin': 20_000, # number of samples to discard from start of chains 
        'num_proc': args.nproc,
        'polyorder': 1, # order of the polynomial systematics model, recommend 1 as higher orders can dilute transit signal 
        'show_progress_bar': False, # show the MCMC progress bar or not 
        'num_proc_slc': args.nproc, # not used, set independently in stage 5 
        'samples_slc': 1_000 # not really used, set independently in stage 5
    }

    # prepare the data by extracting arrays from .h5 files and computing the expected transit times 
    times, spects, errs, _, specs, flux = prep_data(control_dict)
    
    # compute the initial transit model based on the transit parameters 
    # downloaded from the exoplanet archive, mask out the transit in the data based 
    # on the expected time and duration of the transit 
    initial_models, masks = check_initial_state(control_dict)

    # plot the white light timeseries with initial model and points masked alongside the 
    # time vs. wavelength image for the observation. 
    n_obs = len(times)
    fig, axs = plt.subplots(
        n_obs, 2, 
        figsize=(20, 5), 
        gridspec_kw={'hspace': 0.2, 'wspace': 0.1, 'width_ratios':(0.2, 0.8)}
    )
    if len(axs.shape) == 1:
        axs = axs[None, :]
    for i in range(n_obs):

        axs[i, 0].plot(times[i], initial_models[i])
        axs[i, 0].plot(times[i][~masks[i]], flux[i][~masks[i]], 'k.', markersize=0.2)
        axs[i, 0].plot(times[i][masks[i]], flux[i][masks[i]], 'r.', markersize=0.2)
        axs[i, 1].imshow(spects[i] / np.nanmedian(spects[i], axis=0), aspect='auto', vmin=0.99, vmax=1.01)

    plt.savefig(args.out_dir + '/' + args.disp_filt.replace('/', '_') + '_initial_model.pdf')

    # run the MCMC 
    wlc_res = fit(control_dict)
    
    # plot the white lightcurve with samples from the MCMC posterior overplotted
    for i, res in enumerate(wlc_res):
        fig, axs = plt.subplots(
            2, 1, 
            figsize=(15, 4 * n_obs), 
            gridspec_kw={'height_ratios': (0.6, 0.4), 'hspace': 0.0, 'wspace': 0.15}, 
            sharex=True
        )

        axs[0].annotate(args.disp_filt, fontsize=20, xy=(0.05, 0.1), xycoords='axes fraction')

        models = get_model_samples(res, n=100)

        axs[0].plot(
            res['time'], 
            res['flux'], 
            '.', color=plt.cm.rainbow(0.1), markersize=1.0
        )
        [axs[0].plot(res['time'], m, color='k', alpha=0.3) for m in models];
        axs[1].plot(
            res['time'], 
            res['flux'] - np.mean(models, axis=0), 
            '.', color=plt.cm.rainbow(0.1), markersize=1.0
        )

        plt.savefig(args.out_dir + '/' + args.disp_filt.replace('/', '_') + '_white_light_fit_transit{}.pdf'.format(i))
    
    # output the pickle file containing the full MCMC posteriors and control dictionary 
    with open(args.out_dir + '/' + args.disp_filt.replace('/', '_') + '_wl_result.pkl', 'wb') as handle:
        pickle.dump(wlc_res, handle)
        
if __name__ == '__main__':
    sys.exit(main())
