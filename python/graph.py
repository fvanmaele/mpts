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

from trial import run_trial
from trial_precond import *


# %%
def run_trial_precond(mtx, xs, k_max_outer=10, k_max_inner=20, title=None, title_xs=None, custom=None):
    """ Compare the performance of spanning tree preconditioners
    """
    sc, sc_mst, sc_lf = [], [], []
    sd, sd_mst, sd_lf = [], [], []
    labels, labels_mst, labels_lf = [], [], []
    preconds, preconds_mst, preconds_lf = [], [], []

    # Unpreconditioned system
    orig = precond_orig(mtx)
    sc.append(1)
    sd.append(orig['s_degree'])
    preconds.append(None)
    labels.append('unpreconditioned')    

    # # Jacobi preconditioner
    # jacobi = precond_jacobi(mtx)
    # sc.append(jacobi['s_coverage'])
    # sd.append(jacobi['s_degree'])
    # preconds.append(jacobi['precond'])
    # labels.append('jacobi')

    # # Tridiagonal preconditioner
    # tridiag = precond_tridiag(mtx)
    # sc.append(tridiag['s_coverage'])
    # sd.append(tridiag['s_degree'])
    # preconds.append(tridiag['precond'])
    # labels.append('tridiag')

    # Maximum spanning tree preconditioner
    max_st = precond_max_st(mtx)
    sc_mst.append(max_st['s_coverage'])
    sd_mst.append(max_st['s_degree'])
    preconds_mst.append(max_st['precond'])
    labels_mst.append('maxST')
    
    # # Maximum spanning tree preconditioner, applied to pruned matrix
    # max_st_pruned = precond_max_st(mtx, q_max=20)
    # sc_mst.append(max_st_pruned['s_coverage'])
    # sd_mst.append(max_st_pruned['s_degree'])
    # preconds_mst.append(max_st_pruned['precond'])
    # labels_mst.append('maxST (pruned A)')

    # Maximum spanning tree preconditioner, additive factors (m = 2, 3, 4)
    for m in [2, 3, 4]:
        max_st_add_m = precond_max_st_add_m(mtx, m)
        sc_mst.append(max_st_add_m['s_coverage'])
        sd_mst.append(max_st_add_m['s_degree'])
        preconds_mst.append(max_st_add_m['precond'])
        labels_mst.append(f'maxST+ (m = {m})')

    # # Maximum spanning tree preconditioner, MOS-a factors (m = 2, 3, 4)
    # for m in [2, 3, 4]:
    #     max_st_mos_m = precond_max_st_mos_m(mtx, m)
    #     sc_mst.append(max_st_mos_m['s_coverage'])
    #     sd_mst.append(max_st_mos_m['s_degree'])
    #     preconds_mst.append(max_st_mos_m['precond'])
    #     labels_mst.append(f'maxST*a (m = {m})')

    # # Maximum spanning tree preconditioner, MOS-d factors (m = 2, 3, 4)
    # for m in [2, 3, 4]:
    #     max_st_mos_d = precond_max_st_mos_d(mtx, m, scale=0)
    #     sc_mst.append(max_st_mos_d['s_coverage'])
    #     sd_mst.append(max_st_mos_d['s_degree'])
    #     preconds_mst.append(max_st_mos_d['precond'])
    #     labels_mst.append(f'maxST*d (m = {m}')

    # # Maximum spanning tree preconditioner, inner alternating factors (m = 2, 3, 4)
    # for m in [2, 3, 4]:
    #     max_st_alt_m_i = precond_max_st_alt_i(mtx, m, scale=0.01)
    #     sc_mst.append(max_st_alt_m_i['s_coverage'])
    #     sd_mst.append(max_st_alt_m_i['s_degree'])
    #     preconds_mst.append(max_st_alt_m_i['precond'])
    #     labels_mst.append(f'maxST_alt_i (m = {m})')

    # # Maximum spanning tree preconditioner, outer alternating factors (m = 2, 3, 4)
    # for m in [2, 3, 4]:
    #     max_st_alt_m_o = precond_max_st_alt_o(mtx, m, scale=0.01, repeat_i=0)
    #     sc_mst.append(max_st_alt_m_o['s_coverage'])
    #     sd_mst.append(max_st_alt_m_o['s_degree'])
    #     preconds_mst.append(max_st_alt_m_o['precond'])
    #     labels_mst.append(f'maxST_alt_o (m = {m})')

    # Maximum linear forest preconditioner
    max_lf = precond_max_lf(mtx)
    sc_lf.append(max_lf['s_coverage'])
    sd_lf.append(max_lf['s_degree'])
    preconds_lf.append(max_lf['precond'])
    labels_lf.append('maxLF')
    
    # Maximum linear forest preconditioner, additive factors (m = 2, 3, 4)
    for m in [2, 3, 4]:
        max_lf_add_m = precond_max_lf_add_m(mtx, m)
        sc_lf.append(max_lf_add_m['s_coverage'])
        sd_lf.append(max_lf_add_m['s_degree'])
        preconds_lf.append(max_lf_add_m['precond'])
        labels_lf.append(f'maxLF+ (m = {m})')

    # # Maximum linear forest preconditioner, MOS-a factors (m = 2, 3, 4)
    # for m in [2, 3, 4]:
    #     max_lf_mos_m = precond_max_lf_mos_m(mtx, m)
    #     sc_lf.append(max_lf_mos_m['s_coverage'])
    #     sd_lf.append(max_lf_mos_m['s_degree'])
    #     preconds_lf.append(max_lf_mos_m['precond'])
    #     labels_lf.append(f'maxLF*a (m = {m})')

    # # Maximum linear forest preconditioner, MOS-d factors (m = 2, 3, 4)
    # for m in [2, 3, 4]:
    #     max_lf_mos_d = precond_max_lf_mos_d(mtx, m, scale=0)
    #     sc_lf.append(max_lf_mos_d['s_coverage'])
    #     sd_lf.append(max_lf_mos_d['s_degree'])
    #     preconds_lf.append(max_lf_mos_d['precond'])
    #     labels_lf.append(f'maxLF*d (m = {m}')
    
    # # Maximum linear forest preconditioner, inner alternating factors (m = 2, 3, 4)
    # for m in [2, 3, 4]:
    #     max_lf_alt_m_i = precond_max_lf_alt_i(mtx, m, scale=0.01)
    #     sc_lf.append(max_lf_alt_m_i['s_coverage'])
    #     sd_lf.append(max_lf_alt_m_i['s_degree'])
    #     preconds_lf.append(max_lf_alt_m_i['precond'])
    #     labels_lf.append(f'maxLF_alt_i (m = {m})')

    # # Maximum linear forest preconditioner, outer alternating factors (m = 2, 3, 4)
    # for m in [2, 3, 4]:
    #     max_lf_alt_m_o = precond_max_lf_alt_o(mtx, m, scale=0.01)
    #     sc_lf.append(max_lf_alt_m_o['s_coverage'])
    #     sd_lf.append(max_lf_alt_m_o['s_degree'])
    #     preconds_lf.append(max_lf_alt_m_o['precond'])
    #     labels_lf.append(f'maxLF_alt_o (m = {m})')
    
    # # Maximum linear forest preconditioner, MOS-d factors (m = 2, 3, 4)
    # for m in [2, 3, 4]:
    #     max_lf_mos_d = precond_max_lf_mos_d(mtx, m, scale=0.01)
    #     sc_lf.append(max_lf_mos_d['s_coverage'])
    #     sd_lf.append(max_lf_mos_d['s_degree'])
    #     preconds_lf.append(max_lf_mos_d['precond'])
    #     labels_lf.append(f'maxLF*d (m = {m}')

    # iLU(0)
    ilu0 = precond_ilu0(mtx)
    sc.append(None)
    sd.append(None)
    preconds.append(ilu0['precond'])
    labels.append('iLU')

    for xi, x in enumerate(xs):
        title_x = title_xs[xi]

        # Use logarithmic scale for relative residual (y-scale)
        fig1, ax1 = plt.subplots(nrows=1, ncols=2, sharey=True)
        fig1.set_size_inches(10, 7)
        fig1.set_dpi(300)
    
        ax1[0].set_yscale('log')
        ax1[0].set_xlabel('iterations')
        ax1[0].set_ylabel('relres')
    
        ax1[1].set_yscale('log')
        ax1[1].set_xlabel('iterations')
        ax1[1].set_ylabel('relres')
        
        # Classical preconditioners
        for i, label in enumerate(labels):
            M = preconds[i]
    
            if M is None and i > 0:
                print("{}, s_coverage: {}, s_degree: {}".format(label, sc[i], sd[i]))
                continue
            
            result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
        
            if result is not None:
                relres = result['rk']
                fre    = result['fre']
        
                print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                    label, result['iters'], sc[i], sd[i], relres[-1], fre[-1]))
                
                # Results included in both MST/LF graphs
                ax1[0].plot(range(1, len(relres)+1), relres, label=label)
                ax1[1].plot(range(1, len(relres)+1), relres, label=label)
            else:
                warnings.warn(f'failed to solve {label} system')
    
    
        # MST preconditioners
        for i, label in enumerate(labels_mst):
            M = preconds_mst[i]
    
            result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
    
            if result is not None:
                relres = result['rk']
                fre    = result['fre']
        
                print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                    label, result['iters'], sc_mst[i], sd_mst[i], relres[-1], fre[-1]))
        
                # Plot results for MST preconditioner
                ax1[0].plot(range(1, len(relres)+1), relres, label=label)
            else:
                warnings.warn(f'failed to solve {label} system')
        
        
        # LF preconditioners
        for i, label in enumerate(labels_lf):
            M = preconds_lf[i]
    
            result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
    
            if result is not None:
                relres = result['rk']
                fre    = result['fre']
        
                print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                    label, result['iters'], sc_lf[i], sd_lf[i], relres[-1], fre[-1]))
    
                # Plot results for LF preconditioner
                ax1[1].plot(range(1, len(relres)+1), relres, label=label)
            else:
                warnings.warn(f'failed to solve {label} system')
    
    
        # Save picture
        ax1[0].legend(title=f'{title}, x{title_x}')
        ax1[1].legend(title=f'{title}, x{title_x}')
        fig1.savefig(f'{title}_x{title_x}.png', bbox_inches='tight')
    
        plt.close()


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
