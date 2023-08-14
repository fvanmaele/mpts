#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fvanmaele
"""

from pathlib import Path
from scipy.io import mmread

import warnings
import numpy as np
import matplotlib.pyplot as plt
import json

from trial import run_trial
from graph_setup import setup_precond, setup_precond_mst, setup_precond_lf


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# %%
def run_trial_precond(mtx, xs, k_max_outer=10, k_max_inner=20, title=None, title_xs=None):    
    # Generate plots for different right-hand sides
    # TODO: save all data as JSON files, do plotting in different file (easier adjustment and regeneration of plots)
    for xi, x in enumerate(xs):
        title_x = title_xs[xi]
        nrows, ncols = 7, 2

        # Use logarithmic scale for relative residual (y-scale)
        fig1, ax1 = plt.subplots(nrows=nrows, ncols=ncols, sharey=True)
        fig1.set_size_inches(10, 26)
        fig1.set_dpi(300)
    
        for i in range(nrows):
            ax1[i, 0].set_yscale('log')
            ax1[i, 0].set_xlabel('iterations')
            ax1[i, 0].set_ylabel('relres')
        
            ax1[i, 1].set_yscale('log')
            ax1[i, 1].set_xlabel('iterations')
            ax1[i, 1].set_ylabel('relres')
        
        
        # Classical preconditioners
        ref_trials = ['orig', 'jacobi', 'tridiag', 'ilu0']
        
        for i, label in enumerate(ref_trials):
            M  = preconds[label]['precond']
            sc = preconds[label]['s_coverage']
            sd = preconds[label]['s_degree']

            if M is None and i > 0:
                print("{}, s_coverage: {}, s_degree: {}".format(label, sc, sd))
                continue
            
            result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
        
            if result is not None:
                relres = result['rk']
                fre    = result['fre']
        
                print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                    label, result['iters'], sc, sd, relres[-1], fre[-1]))
                
                ax1[0, 0].plot(range(1, len(relres)+1), relres, label=label)
                ax1[0, 1].plot(range(1, len(relres)+1), relres, label=label)
            else:
                warnings.warn(f'failed to solve {label} system')
    
    
        # MST preconditioner
        M  = preconds['max_st']['precond']
        sc = preconds['max_st']['s_coverage']
        sd = preconds['max_st']['s_degree']
        
        if M is None and i > 0:
            print("{}, s_coverage: {}, s_degree: {}".format(label, sc, sd))
            continue
        
        result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
        
        if result is not None:
            relres = result['rk']
            fre    = result['fre']
        
            print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                label, result['iters'], sc, sd, relres[-1], fre[-1]))
            
            # Include basic MST results in every MST row
            for i in range(1, nrows):
                ax1[i, 0].plot(range(1, len(relres)+1), relres, label=label)
        else:
            warnings.warn(f'failed to solve {label} system')


        # MST + factor preconditioners
        mst_trials = ['max_st_add',   'max_st_mos_a', 'max_st_mos_d', 
                      'max_st_alt_i', 'max_st_alt_o', 'max_st_alt_o_repeat']
        
        for i, label in enumerate(mst_trials, start=1):
            for m in range(len(preconds[label])):
                M  = preconds[label][m]['precond']
                sc = preconds[label][m]['s_coverage']
                sd = preconds[label][m]['s_degree']
                
                if M is None and i > 0:
                    print("{}, s_coverage: {}, s_degree: {}".format(label, sc, sd))
                    continue
                
                result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
                
                if result is not None:
                    relres = result['rk']
                    fre    = result['fre']
                
                    print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                        label, result['iters'], sc, sd, relres[-1], fre[-1]))
                    
                    ax1[i, 0].plot(range(1, len(relres)+1), relres, label=label)
                else:
                    warnings.warn(f'failed to solve {label} system')
            

        # LF preconditioner
        M  = preconds['max_lf']['precond']
        sc = preconds['max_lf']['s_coverage']
        sd = preconds['max_lf']['s_degree']
        
        if M is None and i > 0:
            print("{}, s_coverage: {}, s_degree: {}".format(label, sc, sd))
            continue
        
        result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
        
        if result is not None:
            relres = result['rk']
            fre    = result['fre']
        
            print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                label, result['iters'], sc, sd, relres[-1], fre[-1]))
            
            # Include basic LF results in every LF row
            for i in range(1, nrows):
                ax1[i, 1].plot(range(1, len(relres)+1), relres, label=label)
        else:
            warnings.warn(f'failed to solve {label} system')


        # LF + factor preconditioners
        lf_trials = ['max_lf_add',   'max_lf_mos_a', 'max_lf_mos_d', 
                     'max_lf_alt_i', 'max_lf_alt_o', 'max_lf_alt_o_repeat']
        
        for i, label in enumerate(lf_trials, start=1):
            for m in range(len(preconds[label])):
                M  = preconds[label][m]['precond']
                sc = preconds[label][m]['s_coverage']
                sd = preconds[label][m]['s_degree']

                if M is None and i > 0:
                    print("{}, s_coverage: {}, s_degree: {}".format(label, sc, sd))
                    continue
                
                result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
                
                if result is not None:
                    relres = result['rk']
                    fre    = result['fre']
                
                    print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                        label, result['iters'], sc, sd, relres[-1], fre[-1]))
                    
                    ax1[i, 1].plot(range(1, len(relres)+1), relres, label=label)
                else:
                    warnings.warn(f'failed to solve {label} system')
        
    
        # Save picture
        for i in range(nrows):
            for j in range(ncols):
                ax1[i, j].legend(title=f'{title}, x{title_x}')
        fig1.savefig(f'{title}_x{title_x}.png', bbox_inches='tight')
    
        plt.close()


# %%
def main(mtx_path, seed, max_outer, max_inner):
    np.seterr(all='raise')
    
    np.random.seed(seed)
    mtx   = mmread(mtx_path)
    n, m  = mtx.shape
    title = mtx_path.stem

    # Remove explicit zeros set in some matrices (in-place)
    mtx.eliminate_zeros()

    # Right-hand sides
    x1 = np.random.randn(n, 1)
    x2 = np.ones((n, 1))
    x3 = np.sin(np.linspace(0, 100*np.pi, n))

    print(f'{title}, rhs: normally distributed')
    run_trial_precond(mtx, [x1, x2, x3], k_max_outer=max_outer, k_max_inner=max_inner, 
                      title=title, title_xs=['randn', 'ones', 'sine'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='mst_precond', description='trials for spanning tree preconditioner')
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed for random numbers')
    parser.add_argument('--max-outer', type=int, default=15, 
                        help='maximum number of outer GMRES iterations')
    parser.add_argument('--max-inner', type=int, default=20,
                        help='maximum number of inner GMRES iterations')
    parser.add_argument('mtx', type=str)
    
    args = parser.parse_args()
    main(Path(args.mtx), args.seed, args.max_outer, args.max_inner)
