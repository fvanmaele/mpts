#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:41:25 2023

@author: archie
"""

from scipy import sparse
import ilupp
import numpy as np


def lu_sparse_operator(P, inexact=False):
    """ SuperLU based preconditioner for use with GMRES
    """
    assert sparse.issparse(P)
    nrows, ncols = P.shape

    if inexact:
        return ilupp.ILU0Preconditioner(sparse.csc_matrix(P))
    else:
        return sparse.linalg.LinearOperator((nrows, ncols), sparse.linalg.splu(P).solve)


class AltLinearOperator(sparse.linalg.LinearOperator):
    """ Class for non-stationary preconditioners, used with FGMRES.
    """
    def __init__(self, A, Mi):
        assert len(Mi) > 0

        # LinearOperator
        self.shape = A.shape  # assumed consistent between `Mi`
        self.dtype = None        
        super().__init__(self.dtype, self.shape)
        
        # Child members
        self.i = len(Mi)
        self.Mi = Mi
        self.iter = 0       # current iteration, taken modulo `i`

    def _matvec(self, x):
        # Select preconditioner (inverse) based on current iteration
        M = self.Mi[self.iter % self.i]
        self.iter += 1

        # Support objects with `.solve` method and (sparse) matrices
        return M(x) if callable(M) else M @ x


class IterLinearOperator(sparse.linalg.LinearOperator):
    """ Class for iterative refinement with a series of preconditioners
    """
    def __init__(self, A, Mi, repeat_i=0):
        assert len(Mi) > 0
        assert sparse.issparse(A)

        # Iterative refinement with single preconditioner
        if repeat_i > 0:
            assert len(Mi) == 1

        # LinearOperator
        self.shape = A.shape
        self.dtype = None
        super().__init__(self.dtype, self.shape)

        # Child members
        self.A = A
        if repeat_i > 0:
            self.Mi = Mi * repeat_i
        else:
            self.Mi = Mi

    def _matvec(self, x):
        # Initial estimate
        xk = np.zeros_like(x)

        # Loop over series of preconditioners
        for M in self.Mi:
            if callable(M):
                xk += M(x - self.A @ xk)
            else:
                xk += M @ (x - self.A @ xk)

        return xk
