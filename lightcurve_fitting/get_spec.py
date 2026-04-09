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

    args = cli.parse_args()

    if args.out_dir == '':
        args.out_dir = args.data_dir
        
    files = glob.glob(args.data_dir + '/*spec_result.pkl')

    samples = 10000
    
    slc_results = []
    for file in files:
        with open(file, 'rb') as f:
            slc_results.append(pickle.load(f))
   
    for i, planet in enumerate(slc_results[0]['control_dict']['planets']):
        plt.figure(figsize=(8, 4))
        binned_wavs = np.concatenate([res['wavs'] for res in slc_results])
        depths = np.array([np.median(p['r{0}'.format(i)][samples//4:]) for p in np.concatenate([res['posteriors'] for res in slc_results])])
        errs = np.array([np.std(p['r{0}'.format(i)][samples//4:]) for p in np.concatenate([res['posteriors'] for res in slc_results])])

        plt.plot(binned_wavs, depths**2 * 1e6, 'ko')
        plt.errorbar(binned_wavs, depths**2 * 1e6, yerr=errs * 2 * depths * 1e6, ls='none', color='k')
        plt.xlabel('wavelength (microns)')
        plt.ylabel('transit depth (ppm)')
        plt.savefig(args.out_dir + planet + '_spectrum.pdf')

        np.savetxt(args.out_dir + planet + '_spec.txt', np.vstack([binned_wavs, depths, errs]).T)
        
if __name__ == '__main__':
    sys.exit(main())
