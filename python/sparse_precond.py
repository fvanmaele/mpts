#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:48:03 2023

@author: archie
"""

import numpy as np
from scipy import sparse

import graph_precond as gp
from sparse_util import s_coverage, s_degree, prune_sparse_matrix
from sparse_lops import lu_sparse_operator, AltLinearOperator, IterLinearOperator


# %%
def precond_orig(mtx):
    return {
        's_coverage': 1,
        's_degree'  : s_degree(mtx),
        'precond'   : None
    }

def precond_jacobi(mtx):
    P = mtx.diagonal()

    try:
        M = sparse.diags(1. / P, format='csc')
    except FloatingPointError:
        M = None

    return { 
        's_coverage': s_coverage(mtx, sparse.diags(P)),
        's_degree'  : s_degree(sparse.diags(P)),
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

def precond_max_st(mtx, mtx_is_symmetric, prune=None):
    if not mtx_is_symmetric:
        P = gp.spanning_tree_precond(mtx, symmetrize=True)
    else:
        P = gp.spanning_tree_precond(mtx)

    # LU (with the correct permutation) applied to a spanning tree has no fill-in.
    # TODO: factorize the spanning tree conditioner "layer by layer"
    try:
        if prune:
            P = prune_sparse_matrix(P.tolil(), prune)
        M = lu_sparse_operator(P)
    except RuntimeError:
        M = None

    return { 
        's_coverage': s_coverage(mtx, P),
        's_degree'  : s_degree(P),
        'precond'   : M
    }

def precond_max_st_add_m(mtx, mtx_is_symmetric, m):
    assert m > 1
    
    if not mtx_is_symmetric:
        P = gp.spanning_tree_precond_add_m(mtx, m, symmetrize=True)
    else:
        P = gp.spanning_tree_precond_add_m(mtx, m)

    # Accumulation of spanning tree factors may result in any amount of cycles.
    # Use sparse LU decomposition and hope for the best
    try:
        M = lu_sparse_operator(P)
    except RuntimeError:
        M = None

    return { 
        's_coverage': s_coverage(mtx, P),
        's_degree'  : s_degree(P),
        'precond'   : M
    }

def precond_max_st_alt_i(mtx, mtx_is_symmetric, m, scale):
    assert m > 1

    if not mtx_is_symmetric:
        Pi = gp.spanning_tree_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = gp.spanning_tree_precond_list_m(mtx, m, scale=scale)

    try:
        M = AltLinearOperator(mtx.shape, [sparse.linalg.splu(P).solve for P in Pi])

    except RuntimeError:
        M = None

    return {
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }

def precond_max_st_alt_o(mtx, mtx_is_symmetric, m, scale, repeat_i=0):
    if m == 1:
        assert repeat_i > 0

    if not mtx_is_symmetric:
        Pi = gp.spanning_tree_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = gp.spanning_tree_precond_list_m(mtx, m, scale=scale)

    try:
        M = IterLinearOperator(mtx, [sparse.linalg.splu(P).solve for P in Pi], repeat_i=repeat_i)

    except RuntimeError:
        M = None

    return {
            's_coverage': None,
            's_degree'  : None,
            'precond'   : M
        }

def precond_max_st_mos_m(mtx, mtx_is_symmetric, m, prune=None):
    symmetrize = not mtx_is_symmetric
        
    try:
        P_list, B_diff = gp.spanning_tree_precond_mos_m(mtx, m, symmetrize=symmetrize)
        P_list.reverse()
        P = P_list.pop()

        for Pl in P_list:  # XXX: can be done in log_2(n) iterations
            P = Pl @ P
        print(f'{m}: {np.max(B_diff)}')

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
            's_degree'  : s_degree(P),
            'precond'   : M
        }
    else:
        return {
            's_coverage': None,
            's_degree'  : None,
            'precond'   : None
        }

def precond_max_lf(mtx, mtx_is_symmetric):
    if not mtx_is_symmetric:
        P = gp.linear_forest_precond(mtx, symmetrize=True)
    else:
        P = gp.linear_forest_precond(mtx)

    # Note: not permuted to tridiagonal system (tridiagonal solver)
    try:
        M = lu_sparse_operator(P)
    except RuntimeError:
        M = None

    return { 
        's_coverage': s_coverage(mtx, P),
        's_degree'  : s_degree(P),
        'precond'   : M
    }

def precond_max_lf_add_m(mtx, mtx_is_symmetric, m):
    assert m > 1
    
    if not mtx_is_symmetric:
        P = gp.linear_forest_precond_add_m(mtx, m, symmetrize=True)
    else:
        P = gp.linear_forest_precond_add_m(mtx, m)

    # Accumulation of (after permutation) tridiagonal factors
    try:
        M = lu_sparse_operator(P)
    except RuntimeError:
        M = None

    return { 
        's_coverage': s_coverage(mtx, P),
        's_degree'  : s_degree(P),
        'precond'   : M
    }

def precond_max_lf_alt_i(mtx, mtx_is_symmetric, m, scale):
    assert m > 1

    if not mtx_is_symmetric:
        Pi = gp.linear_forest_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = gp.linear_forest_precond_list_m(mtx, m, scale=scale)
    
    try:
        M = AltLinearOperator(mtx.shape, [sparse.linalg.splu(P).solve for P in Pi])

    except RuntimeError:
        M = None

    return {
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }

def precond_max_lf_alt_o(mtx, mtx_is_symmetric, m, scale, repeat_i=0):
    if m == 1:
        assert repeat_i > 0

    if not mtx_is_symmetric:
        Pi = gp.linear_forest_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = gp.linear_forest_precond_list_m(mtx, m, scale=scale)

    try:
        M = IterLinearOperator(mtx, [sparse.linalg.splu(P).solve for P in Pi], repeat_i=repeat_i)

    except RuntimeError:
        M = None

    return {
            's_coverage': None,
            's_degree'  : None,
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

def precond_custom(mtx, P):
    try:
        M = lu_sparse_operator(P)
    except RuntimeError:
        M = None

    return {
        's_coverage': None,
        's_degree'  : s_degree(P),
        'precond'   : M
    }
