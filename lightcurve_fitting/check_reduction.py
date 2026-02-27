import os
os.environ['EXO_LD_PATH'] = '/Users/tylergordon/research/exotic_ld_data'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append('../get_planet_params/')
from fit_wlc import check_initial_state, prep_data

files = glob.glob('*/*/*/stage3/*/*/*.h5')

for file in files:
    
    if 'nrs1' in file:
        detector = 'nrs1'
    elif 'nrs2' in file:
        detector = 'nrs2'
    else:
        raise Error('file is not a g395h reduction ')

    target = files[0].split('/')[-7].replace('_', ' ')
    program, visit = files[0].split('/')[-6].split('_')
    info_string = '{}\ntarget: {}\nvisit: {}'.format(target, program, visit)
    reduction_string = 'extraction ap width: {}\nbackground ap width: {}'.format(ap, bg)
    
    control_dict = {
        'pipeline': 'eureka',
        'data_directories': [file],
        'detector': 'nrs2',
        'polyorder': 1,
        'columns_to_mask': []
    }
    
    time, spect, err, _, spec, flux, _ = prep_data(control_dict)
    time = time[0]
    spect = spect[0]
    err = err[0]
    spec = spec[0]
    flux = flux[0]
    
    mask = build_mask(flux)
    coeffs, fit = gls_fit(
        time, 
        flux, 
        [], 
        mask, 
        polyorder=1, 
        return_coeffs=True
    )

    med = gaussian_filter1d(flux, 20)
    mad = np.median(np.abs(flux - med)) / coeffs[0] * 1e6

    fig, axs = plt.subplots(1, 2, figsize=(25, 5), gridspec_kw={'hspace': 0.2, 'wspace': 0.1, 'width_ratios':(0.3, 0.7)})

    axs[0].plot(time, flux, 'k.', markersize=0.5)
    axs[1].imshow(spect / np.nanmedian(spect, axis=0), aspect='auto', vmin=0.99, vmax=1.01)
    axs[0].annotate(info_string, xy=(0.05, 0.1), xycoords='axes fraction', fontsize=15)
    axs[0].annotate(reduction_string, xy=(0.45, 0.1), xycoords='axes fraction', fontsize=15)
    axs[0].annotate('MAD={:0.2f} ppm'.format(mad), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=15)
    
    h = np.max(flux) - np.min(flux)
    axs[0].set_ylim(np.min(flux) - h / 2, np.max(flux) + h / 10)
    axs[0].set_xlabel('time (days)', fontsize=15)
    axs[0].set_ylabel('flux (counts)', fontsize=15)
    axs[1].set_xlabel('pixel', fontsize=15)
    axs[1].set_ylabel('time index', fontsize=15)
    
    plt.subplots_adjust(left=0.04, right=0.97)
    plt.savefig('{}_{}_{}_{}.pdf'.format(target, program, visit, detector))