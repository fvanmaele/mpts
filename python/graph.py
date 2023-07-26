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

from sparse_util import sparse_is_symmetric
from sparse_precond import *
from solver import run_trial


# %%
def run_trial_precond(mtx, x, k_max_outer=10, k_max_inner=20, title=None, title_x=None, custom=None):
    """ Compare the performance of spanning tree preconditioners
    """
    mtx_is_symmetric = sparse_is_symmetric(mtx)
    sc = []
    sd = []
    labels = []
    preconds = []

    # Unpreconditioned system
    orig = precond_orig(mtx)
    sc.append(1)
    sd.append(orig['s_degree'])
    preconds.append(None)
    labels.append('unpreconditioned')    

    # Jacobi preconditioner
    jacobi = precond_jacobi(mtx)
    sc.append(jacobi['s_coverage'])
    sd.append(jacobi['s_degree'])
    preconds.append(jacobi['precond'])
    labels.append('jacobi')

    # Tridiagonal preconditioner
    tridiag = precond_tridiag(mtx)
    sc.append(tridiag['s_coverage'])
    sd.append(tridiag['s_degree'])
    preconds.append(tridiag['precond'])
    labels.append('tridiag')

    # Maximum spanning tree preconditioner
    max_st = precond_max_st(mtx, mtx_is_symmetric)
    sc.append(max_st['s_coverage'])
    sd.append(max_st['s_degree'])
    preconds.append(max_st['precond'])
    labels.append('maxST')
    
    # # Maximum spanning tree preconditioner, applied to pruned matrix
    # max_st_pruned = precond_max_st(mtx, mtx_is_symmetric, prune=20)
    # sc.append(max_st_pruned['s_coverage'])
    # sd.append(max_st_pruned['s_degree'])
    # preconds.append(max_st_pruned['precond'])
    # labels.append('maxST (pruned A)')

    # # Maximum spanning tree preconditioner, additive factors (m = 2..5)
    # for m in range(2, 6):
    #     max_st_add_m = precond_max_st_add_m(mtx, mtx_is_symmetric, m)
    #     sc.append(max_st_add_m['s_coverage'])
    #     sd.append(max_st_add_m['s_degree'])
    #     preconds.append(max_st_add_m['precond'])
    #     labels.append(f'maxST+ (m = {m})')

    # Maximum spanning tree preconditioner, MOS factors (m = 2..5)
    for m in range(2, 6):
        max_st_mos_m = precond_max_st_mos_m(mtx, mtx_is_symmetric, m)
        sc.append(max_st_mos_m['s_coverage'])
        sd.append(max_st_mos_m['s_degree'])
        preconds.append(max_st_mos_m['precond'])
        labels.append(f'maxST* (m = {m})')

    # # Maximum spanning tree preconditioner, inner alternating factors (m = 2..5)
    # for m in range(2, 6):
    #     #max_st_alt_m_i = precond_max_st_alt_i(mtx, mtx_is_symmetric, m, scale=0)
    #     max_st_alt_m_i = precond_max_st_alt_i(mtx, mtx_is_symmetric, m, scale=0.01)
    #     sc.append(max_st_alt_m_i['s_coverage'])
    #     sd.append(max_st_alt_m_i['s_degree'])
    #     preconds.append(max_st_alt_m_i['precond'])
    #     labels.append(f'maxST_alt_i (m = {m})')

    # # Maximum spanning tree preconditioner, outer alternating factors (m = 2..5)
    # for m in range(2, 6):
    #     #max_st_alt_m_o = precond_max_st_alt_o(mtx, mtx_is_symmetric, 1, scale=0, repeat_i=m)
    #     max_st_alt_m_o = precond_max_st_alt_o(mtx, mtx_is_symmetric, m, scale=0.01, repeat_i=0)
    #     sc.append(max_st_alt_m_o['s_coverage'])
    #     sd.append(max_st_alt_m_o['s_degree'])
    #     preconds.append(max_st_alt_m_o['precond'])
    #     labels.append(f'maxST_alt_o (m = {m})')

    # Maximum linear forest preconditioner
    max_lf = precond_max_lf(mtx, mtx_is_symmetric)
    sc.append(max_lf['s_coverage'])
    sd.append(max_lf['s_degree'])
    preconds.append(max_lf['precond'])
    labels.append('maxLF')
    
    # Maximum linear forest preconditioner, additive factors (m = 2..5)
    # for m in range(2, 6):
    #     max_lf_add_m = precond_max_lf_add_m(mtx, mtx_is_symmetric, m)
    #     sc.append(max_lf_add_m['s_coverage'])
    #     sd.append(max_lf_add_m['s_degree'])
    #     preconds.append(max_lf_add_m['precond'])
    #     labels.append(f'maxLF+ (m = {m})')

    # # Maximum linear forest preconditioner, MOS factors (m = 2..5)
    # for m in range(2, 6):
    #     max_lf_mult_m = precond_max_lf_mult_m(mtx, mtx_is_symmetric, m, scale=0.01)
    #     sc.append(max_lf_mult_m['s_coverage'])
    #     sd.append(max_lf_mult_m['s_degree'])
    #     preconds.append(max_lf_mult_m['precond'])
    #     labels.append(f'maxLF* (m = {m})')
    
    # # Maximum spanning tree preconditioner, inner alternating factors (m = 2..5)
    # for m in range(2, 6):
    #     max_lf_alt_m_i = precond_max_lf_alt_i(mtx, mtx_is_symmetric, m, scale=0.01)
    #     sc.append(max_lf_alt_m_i['s_coverage'])
    #     sd.append(max_lf_alt_m_i['s_degree'])
    #     preconds.append(max_lf_alt_m_i['precond'])
    #     labels.append(f'maxLF_alt_i (m = {m})')

    # # Maximum spanning tree preconditioner, outer alternating factors (m = 2..5)
    # for m in range(2, 6):
    #     max_lf_alt_m_o = precond_max_lf_alt_o(mtx, mtx_is_symmetric, m, scale=0.01)
    #     sc.append(max_lf_alt_m_o['s_coverage'])
    #     sd.append(max_lf_alt_m_o['s_degree'])
    #     preconds.append(max_lf_alt_m_o['precond'])
    #     labels.append(f'maxLF_alt_o (m = {m})')
    
    # iLU(0)
    ilu0 = precond_ilu0(mtx)
    sc.append(None)
    sd.append(None)
    preconds.append(ilu0['precond'])
    labels.append('iLU')

    # Custom preconditioner    
    if custom is not None:
        custom = precond_custom(mtx, mmread(custom))
        sc.append(None)
        sd.append(custom['s_degree'])
        labels.append('custom')

    # Use logarithmic scale for relative residual (y-scale)
    fig1, ax1 = plt.subplots()    
    fig1.set_size_inches(8, 6)
    fig1.set_dpi(300)

    ax1.set_yscale('log')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('relres')

    # TODO: subplot for relres (left) and forward relative error (right)
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(8, 6)
    fig2.set_dpi(300)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('iterations')
    ax2.set_ylabel('fre')

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
    
            # Plot results for specific preconditioner
            ax1.plot(range(1, len(relres)+1), relres, label=label)
            ax2.plot(range(1, len(fre)+1), fre, label=label)

        else:
            warnings.warn(f'failed to solve {label} system')

    ax1.legend(title=f'{title}, x{title_x}')
    fig1.savefig(f'{title}_x{title_x}.png', bbox_inches='tight')
    
    ax2.legend(title=f'{title}, x{title_x}')
    fig2.savefig(f'{title}_x{title_x}_fre.png', bbox_inches='tight')

    plt.close()
    

def main(mtx_path, seed, max_outer, max_inner, precond=None):
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
    run_trial_precond(mtx, x1, k_max_outer=max_outer, k_max_inner=max_inner, 
                      title=title, title_x='randn', custom=precond)
    
    print(f'\n{title}, rhs: ones')
    run_trial_precond(mtx, x2, k_max_outer=max_outer, k_max_inner=max_inner, 
                      title=title, title_x='ones', custom=precond)
    
    print(f'\n{title}, rhs: sine')
    run_trial_precond(mtx, x3, k_max_outer=max_outer, k_max_inner=max_inner, 
                      title=title, title_x='sine', custom=precond)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='mst_precond', description='trials for spanning tree preconditioner')
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed for random numbers')
    parser.add_argument('--max-outer', type=int, default=15, 
                        help='maximum number of outer GMRES iterations')
    parser.add_argument('--max-inner', type=int, default=20,
                        help='maximum number of inner GMRES iterations')
    parser.add_argument('--precond', type=str, 
                        help='path to preconditioning matrix, to be solved with SuperLU')
    parser.add_argument('mtx', type=str)
    
    args = parser.parse_args()
    main(Path(args.mtx), args.seed, args.max_outer, args.max_inner, args.precond)
