#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 17:04:44 2023

@author: archie
"""

import numpy as np
import precond as pc

from scipy.io    import mmread
from scipy       import sparse
from sparse_util import sparse_prune, sparse_max_n
from precond     import diagp1
from trial       import run_trial


# %%
mtx1 = mmread('../mtx/00_2048/cryg2500.mtx')
#assert mtx1_is_symmetric
Id1 = sparse.eye(mtx1.shape[0])
mtx1_row_idx, _, _ = sparse.find(mtx1)
_, mtx1_q = np.unique(mtx1_row_idx, return_counts=True)

# %% MST (m = 1)
B0 = mtx1.copy()
M0 = pc.spanning_tree_precond(B0)  # includes diagonal of mtx1 (S^diag)
B1 = sparse_prune((B0 @ sparse.linalg.inv(M0)).tocoo(), sparse_max_n(B0, mtx1_q))
B0_diff = sparse.linalg.norm(Id1 - B0)
B1_diff = sparse.linalg.norm(Id1 - B1)

# %% MST/MOS (m = 2)
M1 = pc.spanning_tree_precond(B1)
B2 = sparse_prune((B1 @ sparse.linalg.inv(M1)).tocoo(), sparse_max_n(B1, mtx1_q))
B2_diff = sparse.linalg.norm(Id1 - B2)

# %% MST/MOS (m = 3)
M2 = pc.spanning_tree_precond(B2)
B3 = sparse_prune((B2 @ sparse.linalg.inv(M2)).tocoo(), sparse_max_n(B2, mtx1_q))
B3_diff = sparse.linalg.norm(Id1 - B3)

# %%
T0 = sparse.diags(diagp1(B0))
T1 = sparse.diags(diagp1(B1))
T2 = sparse.diags(diagp1(B2))
T3 = sparse.diags(diagp1(B3))

# %%
precond1_diff = sparse.linalg.norm(mtx1 - B1 @ M0)
precond2_diff = sparse.linalg.norm(mtx1 - B2 @ M1 @ M0)
precond3_diff = sparse.linalg.norm(mtx1 - B3 @ M2 @ M1 @ M0)

# %%
x1 = np.random.randn(mtx1.shape[0], 1)

# %% Preconditioner (m = 1)
n, _ = M0.shape
M_MST = sparse.linalg.LinearOperator(M0.shape, sparse.linalg.splu(M0).solve)
result = run_trial(mtx1, x1, M_MST, k_max_outer=10, k_max_inner=20)
# TODO: s_degree / s_coverage
print(f"{result['iters']} iters, relres: {result['rk'][-1]}, fre: {result['fre'][-1]}")

# %% Preconditioner (m = 2)
# XXX: worse results, but B_m still converges to identity?
M_MOS_2 = sparse.linalg.LinearOperator((n, n), sparse.linalg.splu(T2 @ M1 @ M0).solve)
#M_MOS_2 = sparse.linalg.inv(M0) @ sparse.linalg.inv(M1) @ sparse.linalg.inv(T2)
result = run_trial(mtx1, x1, M_MOS_2, k_max_outer=10, k_max_inner=20)
print(f"{result['iters']} iters, relres: {result['rk'][-1]}, fre: {result['fre'][-1]}")

# %% Preconditioner (m = 3)
# XXX: worse results, but B_m still converges to identity?
M_MOS_3 = sparse.linalg.LinearOperator((n, n), sparse.linalg.splu(T3 @ M2 @ M1 @ M0).solve)
result = run_trial(mtx1, x1, M_MOS_3, k_max_outer=10, k_max_inner=20)
print(f"{result['iters']} iters, relres: {result['rk'][-1]}, fre: {result['fre'][-1]}")
