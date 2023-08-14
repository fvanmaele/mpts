#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:53:51 2023

@author: archie
"""

from pyamg import krylov
import numpy as np


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
    rhs = mtx * x
    counter = gmres_counter()
    residuals = []  # input vector for fgmres residuals

    try:
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

    except ValueError:
        return None
