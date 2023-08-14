#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:48:03 2023

@author: archie
"""

from scipy import sparse

from sparse_util import s_coverage, s_degree, sparse_max_n
from sparse_lops import lu_sparse_operator


# %%
def precond_orig(mtx):
    return {
        's_coverage': 1,
        's_degree'  : s_degree(mtx),
        'precond'   : None
    }


def precond_diag(mtx, diag):
    try:
        M = sparse.diags(1. / diag, format='csc')
    except FloatingPointError:
        M = None

    return { 
        's_coverage': s_coverage(mtx, sparse.diags(diag)),
        's_degree'  : s_degree(sparse.diags(diag)),
        'precond'   : M
    }


def precond_tridiag(mtx):
    diagonals = [mtx.diagonal(k=-1), 
                 mtx.diagonal(k=0), 
                 mtx.diagonal(k=1)]
    P = sparse.diags(diagonals, [-1, 0, 1])
    
    try:
        M = lu_sparse_operator(P)
    except RuntimeError:
        M = None
        
    return { 
        's_coverage': s_coverage(mtx, P),
        's_degree'  : s_degree(P),
        'precond'   : M
    }


def precond_mtx(mtx, P, q_max=None):
    q = [q_max] * mtx.shape[0]
    
    try:
        if q is not None:
            P = sparse_max_n(P.tolil(), q)
        M = lu_sparse_operator(P)
    
    except RuntimeError:
        M = None

    return { 
        's_coverage': s_coverage(mtx, P),
        's_degree'  : s_degree(P),
        'precond'   : M
    }


def precond_lops(mtx, Pi, lop, **kwargs):
    try:
        M = lop(mtx, [sparse.linalg.splu(P).solve for P in Pi], **kwargs)

    except RuntimeError:
        M = None

    return {
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


def precond_prod_r(mtx, Pi):
    try:
        P = Pi.pop()

        for factor in Pi:  # XXX: can be done in log_2(n) iterations
            P = P @ factor

    except RuntimeError:
        P = None

    if P is not None:
        try:
            # TODO: can be implemented as LU for every factor instead of (dense) product
            M = lu_sparse_operator(P)
        except RuntimeError:
            M = None

    return {
        's_coverage': None,
        's_degree'  : s_degree(P) if P is not None else None,
        'precond'   : M
    }


def precond_ilu0(mtx):
    try:
        M = lu_sparse_operator(mtx, inexact=True)
    except RuntimeError:
        M = None
    
    return {
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }

