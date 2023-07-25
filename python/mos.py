#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:04:44 2023

@author: archie
"""

#import warnings
import numpy as np
#import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy import sparse

from sparse_util import sparse_is_symmetric, sparse_mask
from sparse_lops import lu_sparse_operator
from trials import run_trial
import graph_precond as gp


# %% Diagonal preconditioners
def unit(x):
    return np.where(x != 0, x / np.abs(x), 1)

def diagp0(mtx):
    return mtx.diagonal()

def row_sums_excluding_diagonal(mtx):
    # Set the diagonal elements to zero in the CSR matrix
    mtx = mtx.copy()
    mtx.setdiag(0)

    # Compute the sum of row elements excluding the diagonal elements
    return np.array(mtx.sum(axis=1)).flatten()

def diagp1(mtx):
    mtx_d = mtx.diagonal()
    return np.multiply(unit(mtx_d), np.maximum(np.abs(mtx_d), row_sums_excluding_diagonal(abs(mtx))))

def diagp2(mtx):
    mtx_d = mtx.diagonal()
    return np.multiply(unit(mtx_d), np.array(abs(mtx).sum(axis=1)).flatten())

def diagl1(mtx):
    mtx_d = mtx.diagonal()
    return mtx_d + row_sums_excluding_diagonal(abs(mtx))


# %%
mtx1 = mmread('../mtx/00_2048/c-20.mtx')
mtx1_is_symmetric = sparse_is_symmetric(mtx1)
assert mtx1_is_symmetric
Id1 = sparse.eye(mtx1.shape[0])

# %% MST (m = 1)
B0 = mtx1.copy()
M0 = gp.spanning_tree_precond(B0)
B1 = sparse_mask((B0 @ sparse.linalg.inv(M0)).tocoo(), B0.tocoo())
B0_diff = sparse.linalg.norm(Id1 - B0)
B1_diff = sparse.linalg.norm(Id1 - B1)

# %% MST/MOS (m = 2)
M1 = gp.spanning_tree_precond(B1)
B2 = sparse_mask((B1 @ sparse.linalg.inv(M1)).tocoo(), B1.tocoo())
B2_diff = sparse.linalg.norm(Id1 - B2)

# %% MST/MOS (m = 3)
M2 = gp.spanning_tree_precond(B2)
B3 = sparse_mask((B2 @ sparse.linalg.inv(M2)).tocoo(), B2.tocoo())
B3_diff = sparse.linalg.norm(Id1 - B3)

# %%
T0 = sparse.diags(diagp1(B0))
T1 = sparse.diags(diagp1(B1))
T2 = sparse.diags(diagp1(B2))
T3 = sparse.diags(diagp1(B3))

# %%
x1 = np.random.randn(mtx1.shape[0], 1)

# %% Preconditioner (m = 1)
M_MST = lu_sparse_operator(M0)
result = run_trial(mtx1, x1, M_MST, k_max_outer=10, k_max_inner=20)
# TODO: s_degree / s_coverage
print(f"{result['iters']} iters, relres: {result['rk'][-1]}, fre: {result['fre'][-1]}")

# %% Preconditioner (m = 2)
# XXX: worse results, but B_m still converges to identity?
M_MOS_2 = lu_sparse_operator(T2 @ M0 @ M1)
#M_MOS_2 = sparse.linalg.inv(M0) @ sparse.linalg.inv(M1) @ sparse.linalg.inv(T2)
result = run_trial(mtx1, x1, M_MOS_2, k_max_outer=10, k_max_inner=20)
print(f"{result['iters']} iters, relres: {result['rk'][-1]}, fre: {result['fre'][-1]}")

# %% Preconditioner (m = 3)
# XXX: worse results, but B_m still converges to identity?
M_MOS_3 = lu_sparse_operator(T3 @ M0 @ M1 @ M2)
result = run_trial(mtx1, x1, M_MOS_3, k_max_outer=10, k_max_inner=20)
print(f"{result['iters']} iters, relres: {result['rk'][-1]}, fre: {result['fre'][-1]}")

