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
from math import ceil

from sparse_util import sparse_is_symmetric, sparse_mask, prune_sparse_matrix
from sparse_lops import lu_sparse_operator
from sparse_precond import diagp1, diagp0

from trials import run_trial
import graph_precond as gp


# %%
mtx1 = mmread('../mtx/00_2048/c-20.mtx')
mtx1_symmetrize = not sparse_is_symmetric(mtx1)
mtx1_avg_deg = ceil(mtx1.getnnz() / mtx1.shape[0])
#assert mtx1_is_symmetric
Id1 = sparse.eye(mtx1.shape[0])

# %% MST (m = 1)
B0 = mtx1.copy()
M0 = gp.spanning_tree_precond(B0, symmetrize=mtx1_symmetrize)  # includes diagonal of mtx1 (S^diag)
#B1 = sparse_mask((B0 @ sparse.linalg.inv(M0)).tocoo(), B0.tocoo())  # "ILU(0)"
B1 = sparse_mask((B0 @ sparse.linalg.inv(M0)).tocoo(), prune_sparse_matrix(B0 @ B0, mtx1_avg_deg))
#B1 = prune_sparse_matrix(B0 @ sparse.linalg.inv(M0).tolil(), mtx1_avg_deg)
B0_diff = sparse.linalg.norm(Id1 - B0)
B1_diff = sparse.linalg.norm(Id1 - B1)

# %% MST/MOS (m = 2)
M1 = gp.spanning_tree_precond(B1, symmetrize=mtx1_symmetrize)
#B2 = sparse_mask((B1 @ sparse.linalg.inv(M1)).tocoo(), B1.tocoo())
B2 = sparse_mask((B1 @ sparse.linalg.inv(M1)).tocoo(), prune_sparse_matrix(B1 @ B1, mtx1_avg_deg))
#B2 = prune_sparse_matrix(B1 @ sparse.linalg.inv(M1), mtx1_avg_deg)
B2_diff = sparse.linalg.norm(Id1 - B2)

# %% MST/MOS (m = 3)
M2 = gp.spanning_tree_precond(B2, symmetrize=mtx1_symmetrize)
#B3 = sparse_mask((B2 @ sparse.linalg.inv(M2)).tocoo(), B2.tocoo())
B3 = sparse_mask((B2 @ sparse.linalg.inv(M2)).tocoo(), prune_sparse_matrix(B2 @ B2, mtx1_avg_deg))
#B3 = prune_sparse_matrix(B2 @ sparse.linalg.inv(M2), mtx1_avg_deg)
B3_diff = sparse.linalg.norm(Id1 - B3)

# %%
T0 = sparse.diags(diagp0(B0))
T1 = sparse.diags(diagp0(B1))
T2 = sparse.diags(diagp0(B2))
T3 = sparse.diags(diagp0(B3))

# %%
precond1_diff = sparse.linalg.norm(mtx1 - B1 @ M0)
precond2_diff = sparse.linalg.norm(mtx1 - B2 @ M1 @ M0)
precond3_diff = sparse.linalg.norm(mtx1 - B3 @ M2 @ M1 @ M0)

# %%
x1 = np.random.randn(mtx1.shape[0], 1)

# %% Preconditioner (m = 1)
M_MST = lu_sparse_operator(M0)
result = run_trial(mtx1, x1, M_MST, k_max_outer=10, k_max_inner=20)
# TODO: s_degree / s_coverage
print(f"{result['iters']} iters, relres: {result['rk'][-1]}, fre: {result['fre'][-1]}")

# %% Preconditioner (m = 2)
# XXX: worse results, but B_m still converges to identity?
M_MOS_2 = lu_sparse_operator(T2 @ M1 @ M0)
#M_MOS_2 = sparse.linalg.inv(M0) @ sparse.linalg.inv(M1) @ sparse.linalg.inv(T2)
result = run_trial(mtx1, x1, M_MOS_2, k_max_outer=10, k_max_inner=20)
print(f"{result['iters']} iters, relres: {result['rk'][-1]}, fre: {result['fre'][-1]}")

# %% Preconditioner (m = 3)
# XXX: worse results, but B_m still converges to identity?
M_MOS_3 = lu_sparse_operator(T3 @ M2 @ M1 @ M0)
result = run_trial(mtx1, x1, M_MOS_3, k_max_outer=10, k_max_inner=20)
print(f"{result['iters']} iters, relres: {result['rk'][-1]}, fre: {result['fre'][-1]}")

