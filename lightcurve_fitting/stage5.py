import os
import sys
os.environ['EXO_LD_PATH'] = '/data/tagordon/exotic_ld_data'
os.environ['OMP_NUM_THREADS'] = '1'
import pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from fit_slc import fit
import argparse
import glob

def main():

    cli = argparse.ArgumentParser()

    cli.add_argument(
        '--data_dir',
        type = str,
        default = './',
    )

    cli.add_argument(
        '--out_dir',
        type = str,
        default = ''
    )
    
    cli.add_argument(
        '--bins_file',
        type = str,
        default = ''
    )

    cli.add_argument(
        '--nproc',
        type = np.int64,
        default = 1
    )
    
    cli.add_argument(
        '--n_samples',
        type = np.int64,
        default = None
    )
    
    cli.add_argument(
        '--disp_filt',
        type = str,
        default = ''
    )

    args = cli.parse_args()

    if args.out_dir == '':
        args.out_dir = args.data_dir
        
    file = glob.glob(args.data_dir + '/' + args.disp_filt.replace('/', '_') + '_wl_result.pkl')[0]
    with open(file, 'rb') as f:
        res = pickle.load(f)
        
    control_dict = res[0]['control_dict']
        
    if not args.bins_file == '':
        try:
            bin_low, bin_high, _, _ = np.loadtxt(args.bins_file).T
            
            control_dict['bin_low'] = bin_low
            control_dict['bin_high'] = bin_high
        except Exception as e:
            print(e)
            print('Invalid bins file. File should be text file with four columns: bins_short  bins_long  bins_center  bins_width')
            print('Defaulting to 0.02 micron bins.')
            
    if args.n_samples is None:
        samples = control_dict['samples_slc']
    else:
        samples = args.n_samples
   
    if args.nproc > 1:
        control_dict['num_proc_slc'] = args.nproc

    slc_res = fit(res, samples=samples)
     
    plt.figure(figsize=(8, 4))
    binned_wavs = slc_res['wavs']
    depths = np.array([np.median(p['r0'][samples//4:]) for p in slc_res['posteriors']])
    errs = np.array([np.std(p['r0'][samples//4:]) for p in slc_res['posteriors']])

    plt.plot(binned_wavs, depths**2 * 1e6, 'ko')
    plt.errorbar(binned_wavs, depths**2 * 1e6, yerr=errs * 2 * depths * 1e6, ls='none', color='k')
    plt.xlabel('wavelength (microns)')
    plt.ylabel('transit depth (ppm)')
    plt.savefig(args.out_dir + '/spectrum.pdf')
    
    with open(args.out_dir + '/' + args.disp_filt.replace('/', '_') + '_spec_result.pkl', 'wb') as handle:
        pickle.dump(slc_res, handle)

if __name__ == '__main__':
    sys.exit(main())
