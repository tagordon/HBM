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

    cli.add_argument(
        '--planets',
        type = str,
        nargs = '*',
        default = []
    )

    cli.add_argument(
        '--data_dirs',
        type = str,
        nargs = '*',
        default = [],
    )

    cli.add_argument(
        '--out_dir',
        type = str,
        default = []
    )

    cli.add_argument(
        '--planet_refs',
        type = str,
        nargs = '*',
        default = [],
    )

    cli.add_argument(
        '--stellar_ref',
        type = str,
        default = '',
    )

    cli.add_argument(
        '--nproc',
        type = np.int64,
        default = 1
    )

    cli.add_argument(
        '--delta_t0',
        type = np.float64,
        default = 0.0,
    )
    
    cli.add_argument(
        '--disp_filt',
        type = str,
        default = ''
    )

    args = cli.parse_args()

    if args.out_dir == '':
        args.out_dir = args.data_dirs[0]
    if args.stellar_ref == '':
        args.stellar_ref = args.planet_refs[0]
    if len(args.planets) != len(args.planet_refs):
        raise Exception('number of planetary parameter references does not equal number of planets.')
        
    st_name = '_'.join(args.stellar_ref.split('_')[:-1])
    st_year = args.stellar_ref.split('_')[-1]
    priors = []
    for planet, ref in zip(args.planets, args.planet_refs):

        pl_name = '_'.join(ref.split('_')[:-1])
        pl_year = ref.split('_')[-1]

        pl, st_params = load_priors(planet.replace('_', ' '), (pl_name, pl_year), (st_name, st_year))
        priors.append(pl)
        
    if args.disp_filt == '':
        raise Exception('must specifiy disperser/filter combination')
    else:
        start_wav, end_wav = get_wav_ranges(args.disp_filt)

    files = []
    for d in args.data_dirs:
        files.append(glob.glob(d + '/*.h5', recursive=True))
    files = np.concatenate(files)
    print(files, d + '/*.h5')

    control_dict = {
        'pipeline': 'eureka',
        'planets': args.planets,
        'priors': priors,
        'stellar_parameters': st_params,
        'data_directories': files,
        'columns_to_mask': [],
        'delta_t0': args.delta_t0,
        'disp_filt': args.disp_filt,
        'start_wav': start_wav,
        'end_wav': end_wav,
        'samples': 100_000,
        'burnin': 20_000,
        'num_proc': args.nproc,
        'polyorder': 1,
        'show_progress_bar': False,
        'num_proc_slc': args.nproc,
        'samples_slc': 1_000
    }

    times, spects, errs, _, specs, flux = prep_data(control_dict)
    initial_models, masks = check_initial_state(control_dict)

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

    wlc_res = fit(control_dict)
    
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
    
    with open(args.out_dir + '/' + args.disp_filt.replace('/', '_') + '_wl_result.pkl', 'wb') as handle:
        pickle.dump(wlc_res, handle)
        
if __name__ == '__main__':
    sys.exit(main())
