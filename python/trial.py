#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fvanmaele
"""

from pathlib  import Path
from scipy.io import mmread
from scipy    import sparse
from pyamg    import krylov

import warnings
import numpy as np
import json

from trial_setup import precond_setup, precond_setup_mst, precond_setup_lf


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class gmres_counter(object):
    """ Class for counting the number of GMRES iterations (inner+outer)
    """
    def __init__(self):
        self.niter = 0
        self.xk = []

    def __call__(self, xk=None):
        self.niter += 1
        self.xk.append(xk)


def run_trial(mtx, x, M, k_max_outer, k_max_inner):
    """ Solve a (right) preconditioned linear system with a fixed number of GMRES iterations
    """
    # Right-hand side from exact solution
    rhs = mtx @ x
    counter = gmres_counter()
    residuals = []  # input vector for fgmres residuals

    x_gmres, info = krylov.fgmres(mtx, rhs, M=M, x0=None, tol=1e-15, 
                                  restrt=k_max_inner, maxiter=k_max_outer,
                                  callback=counter, residuals=residuals)

    # Normalize to relative residual
    relres = np.array(residuals) / np.linalg.norm(rhs)

    # Compute forward relative error
    x_diff = np.matrix(counter.xk) - x.T
    fre = np.linalg.norm(x_diff, axis=1) / np.linalg.norm(x)

    return {
        #'x':  x,
        'fre': fre.tolist(),
        'rk': relres,
        'exit_code': info, 
        'iters': counter.niter 
    }



def run_trial_precond(mtx, xs, k_max_outer=10, k_max_inner=20, title=None, title_xs=None):    
    preconds_ref = precond_setup(mtx)
    preconds_mst = precond_setup_mst(mtx, 4)
    preconds_lf  = precond_setup_lf(mtx, 4)
    preconds     = {**preconds_ref, **preconds_mst, **preconds_lf}

    # Generate plots for different right-hand sides
    for xi, x in enumerate(xs):
        title_x = title_xs[xi]
        print(f'{title}, x: {title_x}')

        for label, precond_m in preconds.items():
            if len(precond_m) == 1:
                start_m=1
            else:
                start_m=2

            for m, precond in enumerate(precond_m, start=start_m):
                M  = precond['precond']
                sc = precond['s_coverage']
                sd = precond['s_degree']

                if m != 1:
                    label_m = label + f'_m{m}'
                else:
                    label_m = label
                
                if M is None and label != 'orig':
                    print("{}, s_coverage: {}, s_degree: {}".format(label_m, sc, sd))
                    continue
                
                result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)

                if result is not None:
                    result['mtx']    = title
                    result['method'] = label_m
                    result['x_id']   = title_x
                    
                    print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                        label_m, result['iters'], sc, sd, result['rk'][-1], result['fre'][-1]))

                    with open(f'{title}_x{title_x}_{label_m}.json', 'w') as f:
                        json.dump(result, f, cls=NumpyArrayEncoder, indent=2)

                else:
                    warnings.warn(f'failed to solve {label} system with method {label_m}')


# %%
def main(mtx_path, seed, max_outer, max_inner):
    np.seterr(all='raise')
    
    np.random.seed(seed)
    mtx   = sparse.coo_array(mmread(mtx_path))
    n, m  = mtx.shape
    title = mtx_path.stem

    # Remove explicit zeros set in some matrices (in-place)
    mtx.eliminate_zeros()

    # Right-hand sides
    x1 = np.random.randn(n, 1)
    x2 = np.ones((n, 1))
    x3 = np.sin(np.linspace(0, 100*np.pi, n))

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
