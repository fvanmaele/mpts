#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fvanmaele
"""

from pathlib import Path
from scipy.io import mmread

import warnings
import numpy as np
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
        
        preconds_ref = setup_precond(mtx)
        preconds_mst = setup_precond_mst(mtx, 4)
        preconds_lf  = setup_precond_lf(mtx, 4)
        
        preconds = {**preconds_ref, **preconds_mst, **preconds_lf}
        
        for label, precond_m in preconds.items():
            for m, precond in enumerate(precond_m, start=1):
                M  = precond['precond']
                sc = precond['s_coverage']
                sd = precond['s_degree']
                
                if M is None and label != 'orig':
                    print("{}, s_coverage: {}, s_degree: {}".format(label, sc, sd))
                    continue
                
                result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
    
                if result is not None:
                    relres = result['rk']
                    fre    = result['fre']
                    if m > 1:
                        label += f'_m{m}'

                    print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                        label, result['iters'], sc, sd, relres[-1], fre[-1]))
    
                    with open(f'{title}_x{title_x}_{label}.json', 'w') as f:
                        json.dump(result, f, cls=NumpyArrayEncoder)
                else:
                    warnings.warn(f'failed to solve {label} system')


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
