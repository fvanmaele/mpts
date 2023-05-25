#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fvanmaele
"""

# %%
import os

home_dir = os.environ['HOME']
source_dir = '{}/source/repos/arpts'.format(home_dir)
os.chdir('{}/python'.format(source_dir))


# %% Numerical libraries
import numpy as np
import networkx as nx
from scipy.io import mmread
from scipy import sparse
import pylops
import inspect

def cond(A):
    if sparse.issparse(A):
        raise NotImplementedError()
    else:
        return np.format_float_scientific(np.linalg.cond(A))


# TODO: add diagnostics
# XXX: random_spanning_tree does not result in a graph with same amount of nodes as the original graph
def compute_spanning_trees(mtx, n_rand_st=4):
    assert(n_rand_st > 0)
    assert(sparse.issparse(mtx))

    G = nx.from_scipy_sparse_array(mtx)  # COO
    # spanning tree algorithms do not include weights for loops (diagonal 
    # elements in the adjacency graph)
    # retrieve the diagonal for later adding
    D = sparse.diags(mtx.diagonal())     # DIAgonal
    maxST = nx.maximum_spanning_tree(G)
    minST = nx.minimum_spanning_tree(G)

    return {
        'graph': G,
        'max_spanning_tree': maxST, 
        'max_spanning_tree_adj': nx.to_scipy_sparse_array(maxST) + D,  # CSR
        'min_spanning_tree': minST,
        'min_spanning_tree_adj': nx.to_scipy_sparse_array(minST) + D
    }


def ilu_to_linear_operator(iLU):
    nrows, ncols = iLU.shape
    return sparse.linalg.LinearOperator((nrows, ncols), iLU.solve)


# TODO: count matrix vector multiplications in each iteration
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.residuals = []
        self.matvec_counts = []

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))
        self.residuals.append(rk)
        # XXX: limited arguments to callbacks (access to parent linear operator)
        frame = inspect.currentframe().f_back.f_back.f_back.f_back
        #print(frame.f_locals.keys())
        A_parent = frame.f_locals['mtx_op']
        self.matvec_counts.append(A_parent.matvec_count)


""" Solve ILU preconditioned problem, where ILU is applied to a given matrix P
"""
def run_trial(mtx, x, P=None, atol=None, tol=1e-05):
    M = None
    if P is not None:
        M = ilu_to_linear_operator(sparse.linalg.splu(P))
    
    b = mtx * x
    counter = gmres_counter(disp=False)
    # Keep count of matrix/vector multiplications
    mtx_op = pylops.MatrixMult(mtx, dtype='float')
    x_gmres, ec = sparse.linalg.gmres(mtx_op, b, M=M, callback=counter, atol=atol, tol=tol)
    
    fre = np.linalg.norm(x_gmres - x) / np.linalg.norm(x)
    #relres = np.linalg.norm(mtx*x_gmres - b) / np.linalg.norm(b)
    relres = counter.residuals
    #matvec_count = mtx_op.matvec_count
    matvec_count = counter.matvec_counts

    return { 'x': x, 'x_gmres': x_gmres, 'fre': fre, 'relres': relres, 'exit_code': ec, 'iters': counter.niter, 'matvec_count': matvec_count }


# %% Auxiliary libraries
import json
import glob
import matplotlib.pyplot as plt

# %% Real symmetric, positive definite (TODO: diagonally dominant?)
mtx1 = mmread('../mtx/ufsmc/494_bus.mtx')
ST1 = compute_spanning_trees(mtx1)
maxST1_mtx = ST1['max_spanning_tree_adj']
minST1_mtx = ST1['min_spanning_tree_adj']

# %% Real unsymmetric, non positive definite
mtx2 = mmread('../mtx/ufsmc/gre_512.mtx')
ST2 = compute_spanning_trees(mtx2)
maxST2_mtx = ST2['max_spanning_tree_adj']
minST2_mtx = ST2['min_spanning_tree_adj']

# %% Generate right hand sides
np.random.seed(42)
x11 = np.random.randn(494, 1)
x12 = np.ones((494, 1))
x13 = np.sin(np.linspace(0, 100*np.pi, 494))

x21 = np.random.randn(512, 1)
x22 = np.ones((512, 1))
x23 = np.sin(np.linspace(0, 100*np.pi, 512))

# %% GMRES
res11 = run_trial(mtx1, x11)
res11_maxST = run_trial(mtx1, x11, maxST1_mtx)  # iLU(max spanning tree)
res11_minST = run_trial(mtx1, x11, minST1_mtx)  # iLU(min spanning tree)
res11_iLU = run_trial(mtx1, x11, mtx1)          # iLU(A)
print("unpreconditioned: {} iters, maxST: {} iters, minST: {} iters, iLU: {} iters".format(
    res11['iters'], res11_maxST['iters'], res11_minST['iters'], res11_iLU['iters']))

# %%
plt.yscale('log')
plt.plot(res11['matvec_count'], res11['relres'], label='orig')
plt.plot(res11_maxST['matvec_count'], res11_maxST['relres'], label='max_st+ilu')
plt.plot(res11_minST['matvec_count'], res11_minST['relres'], label='min_st+ilu')
plt.plot(res11_iLU['matvec_count'], res11_iLU['relres'], label='ilu')
plt.legend(title='494_bus, x1')

# %%
res12 = run_trial(mtx1, x12)
res12_maxST = run_trial(mtx1, x12, maxST1_mtx)
res12_minST = run_trial(mtx1, x12, minST1_mtx)
res12_iLU = run_trial(mtx1, x12, mtx1)
print("unpreconditioned: {} iters, maxST: {} iters, minST: {} iters, iLU: {} iters".format(
    res12['iters'], res12_maxST['iters'], res12_minST['iters'], res12_iLU['iters']))

# %%
plt.yscale('log')
plt.plot(res12['matvec_count'], res12['relres'], label='orig')
plt.plot(res12_maxST['matvec_count'], res12_maxST['relres'], label='max_st+ilu')
plt.plot(res12_minST['matvec_count'], res12_minST['relres'], label='min_st+ilu')
plt.plot(res12_iLU['matvec_count'], res12_iLU['relres'], label='ilu')
plt.legend(title='494_bus, x2')

# %%
res13 = run_trial(mtx1, x13)
res13_maxST = run_trial(mtx1, x13, maxST1_mtx)
res13_minST = run_trial(mtx1, x13, minST1_mtx)
res13_iLU = run_trial(mtx1, x13, mtx1)
print("unpreconditioned: {} iters, maxST: {} iters, minST: {} iters, iLU: {} iters".format(
    res13['iters'], res13_maxST['iters'], res13_minST['iters'], res13_iLU['iters']))

# %%
plt.yscale('log')
plt.plot(res13['matvec_count'], res13['relres'], label='orig')
plt.plot(res13_maxST['matvec_count'], res13_maxST['relres'], label='max_st+ilu')
plt.plot(res13_minST['matvec_count'], res13_minST['relres'], label='min_st+ilu')
plt.plot(res13_iLU['matvec_count'], res13_iLU['relres'], label='ilu')
plt.legend(title='494_bus, x2')

# %%
res21 = run_trial(mtx2, x21)
res21_maxST = run_trial(mtx2, x21, maxST2_mtx)
res21_minST = run_trial(mtx2, x21, minST2_mtx)
res21_iLU = run_trial(mtx2, x21, mtx2)
print("unpreconditioned: {} iters, maxST: {} iters, minST: {} iters, iLU: {} iters".format(
    res21['iters'], res21_maxST['iters'], res21_minST['iters'], res21_iLU['iters']))

# %%
res22 = run_trial(mtx2, x22)
res22_maxST = run_trial(mtx2, x22, maxST2_mtx)
res22_minST = run_trial(mtx2, x22, minST2_mtx)
res22_iLU = run_trial(mtx2, x22, mtx2)
print("unpreconditioned: {} iters, maxST: {} iters, minST: {} iters, iLU: {} iters".format(
    res22['iters'], res22_maxST['iters'], res22_minST['iters'], res22_iLU['iters']))

# %%
res23 = run_trial(mtx2, x23)
res23_maxST = run_trial(mtx2, x23, maxST2_mtx)
res23_minST = run_trial(mtx2, x23, minST2_mtx)
res23_iLU = run_trial(mtx2, x23, mtx2)
print("unpreconditioned: {} iters, maxST: {} iters, minST: {} iters, iLU: {} iters".format(
    res23['iters'], res23_maxST['iters'], res23_minST['iters'], res23_iLU['iters']))
