#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fvanmaele
"""

# %%
import os
import sys
import warnings

home_dir = os.environ['HOME']
source_dir = '{}/source/repos/pmst'.format(home_dir)
os.chdir('{}/python'.format(source_dir))


# %% Numerical libraries
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.io import mmread
from scipy import sparse
#from copy import copy

#import pylops


# %%
def cond(A):
    if sparse.issparse(A):
        raise NotImplementedError()
    else:
        return np.format_float_scientific(np.linalg.cond(A))


# Check if a matrix is symmetric up to a certain tolerance
# TODO: pattern symmetry vs. numerical symmetry
def sparse_is_symmetric(mtx, tol=1e-10):
    return (abs(mtx-mtx.T) > tol).nnz == 0


# Find roots (no incoming edges) of directed graph
# A maximum spanning arboresence can only be found with an uniquely defined root node
def digraph_find_roots(G):
    assert G.is_directed()
    roots = []

    for deg in G.in_degree:
        if deg[1] == 0:
            roots.append(deg[0])

    return roots, len(roots) == 1


# Find the maximum spanning arboresence for each strongly connected component of a digraph
def digraph_maximum_spanning_forest(G):
    assert G.is_directed()
    
    maximum_spanning_forest = nx.DiGraph()
    strongly_connected_components = nx.strongly_connected_components(G)
    
    for component in strongly_connected_components:
        subgraph = G.subgraph(component)
        maximum_arborescence = nx.maximum_spanning_arborescence(subgraph, attr='weight')
        maximum_spanning_forest = nx.compose(maximum_spanning_forest, maximum_arborescence)

    return maximum_spanning_forest


# Find the minimum spanning arboresence for each strongly connected component of a digraph
def digraph_minimum_spanning_forest(G):
    assert G.is_directed()
    
    minimum_spanning_forest = nx.DiGraph()
    strongly_connected_components = nx.strongly_connected_components(G)
    
    for component in strongly_connected_components:
        subgraph = G.subgraph(component)
        minimum_arborescence = nx.minimum_spanning_arborescence(subgraph, attr='weight')
        minimum_spanning_forest = nx.compose(minimum_spanning_forest, minimum_arborescence)

    return minimum_spanning_forest


# Take all non-zero indices of a sparse matrix `mtx_mask` and apply them to a
# sparse matrix `mtx` of the same dimensions. It is assumed that non-zero indices
# of `mtx_mask` are a subset of non-zero indices of `mtx` (with potentially 
# differing entries)
def sparse_mask(mtx, mtx_mask):
    assert mtx.shape == mtx_mask.shape
    assert sparse.issparse(mtx)
    assert sparse.issparse(mtx_mask)

    rows = []
    cols = []
    data = []
    mask = {}

    for i, j, v in zip(mtx_mask.row, mtx_mask.col, mtx_mask.data):
        mask[(i, j)] = True

    for i, j, v in zip(mtx.row, mtx.col, mtx.data):
        if (i, j) in mask:
            rows.append(i)
            cols.append(j)
            data.append(v)

    return sparse.coo_matrix((data, (rows, cols)))


def compute_spanning_trees(mtx, symmetrize=False):
    assert sparse.issparse(mtx)
    maxST, maxST_pre, minST, minST_pre = None, None, None, None

    # Compute spanning tree on preprocessed matrix. Since we want to compute 
    # spanning trees based on the highest absolute values (weights), 
    # `abs(mtx)` is taken over `mtx` in every case.
    if sparse_is_symmetric(mtx):
        G_abs = nx.Graph(abs(mtx))
        maxST_pre = nx.maximum_spanning_tree(G_abs)  # does not contain diagonal
        minST_pre = nx.minimum_spanning_tree(G_abs)
    
    elif symmetrize:
        G_symm = nx.Graph((abs(mtx) + abs(mtx.T)) / 2)
        maxST_pre = nx.maximum_spanning_tree(G_symm)
        minST_pre = nx.minimum_spanning_tree(G_symm)

    else:
        G_abs = nx.DiGraph(abs(mtx))
        _, has_unique_root = digraph_find_roots(G_abs)

        if has_unique_root:
            maxST_pre = nx.maximum_spanning_arborescence(G_abs)
            minST_pre = nx.minimum_spanning_arborescence(G_abs)
        else:
            maxST_pre = digraph_maximum_spanning_forest(G_abs)
            minST_pre = digraph_minimum_spanning_forest(G_abs)

    # Construct new sparse matrix based on non-zero weights of spanning tree.
    # The diagonal is added manually, because `nx` does not include weights
    # for self-loops.
    D = sparse.diags(mtx.diagonal())     # DIAgonal
    maxST = sparse_mask(mtx, sparse.coo_array(nx.to_scipy_sparse_array(maxST_pre) + D))
    minST = sparse_mask(mtx, sparse.coo_array(nx.to_scipy_sparse_array(minST_pre) + D))

    return {
        'max_spanning_tree': nx.Graph(maxST) if sparse_is_symmetric(maxST) else nx.DiGraph(maxST), 
        'max_spanning_tree_adj': maxST,
        'min_spanning_tree': nx.Graph(minST) if sparse_is_symmetric(minST) else nx.DiGraph(minST),
        'min_spanning_tree_adj': minST
    }


# Compute the S-coverage as quality measure for a sparse preconditioner
def s_coverage(mtx, mtx_pruned):
    assert sparse.issparse(mtx)
    assert sparse.issparse(mtx_pruned)
    
    #sc = sparse.linalg.norm(mtx_pruned, ord=1) / sparse.linalg.norm(mtx, ord=1)
    sc = abs(sparse.csr_matrix(mtx_pruned)).sum() / abs(sparse.csr_matrix(mtx)).sum()
    if sc > 1:
        warnings.warn('S coverage is greater than 1')

    return sc


# Linear operator for preconditioned GMRES
def ilu_to_linear_operator(iLU):
    nrows, ncols = iLU.shape
    return sparse.linalg.LinearOperator((nrows, ncols), iLU.solve)


# Count the number of (inner) GMRES iterations
class gmres_counter(object):
    def __init__(self):
        self.niter = 0
        self.xk = []

    def __call__(self, xk=None):
        self.niter += 1
        self.xk.append(xk)


# %% Solve ILU preconditioned problem, where ILU is applied to a given matrix P
def run_trial(mtx, x, M=None, maxiter=1000, restart=20):
    # Right-hand side from exact solution
    b = mtx * x

    # Keep track of GMRES (inner) iterations
    counter = gmres_counter()
    
    # callback_type='x' to compare FRE in each iteration
    x_gmres, ec = sparse.linalg.gmres(mtx, b, M=M, 
                                      callback=counter, callback_type='x', 
                                      maxiter=maxiter, restart=restart)
    return { 
        'x': x, 
        'x_k': counter.xk, 
        'exit_code': ec, 
        'iters': counter.niter 
    }


# TODO: plot residual
def run_trial_precond(mtx, x, title=None, title_x=None, symmetrize=False):
    # Maximum spanning tree preconditioner
    ST = compute_spanning_trees(mtx, symmetrize=symmetrize)
    P1 = ST['max_spanning_tree_adj']
    P1_sc = s_coverage(mtx, P1)

    # LU (with the correct permutation) applied to a spanning tree does
    # not result in fill-in.
    # TODO: this calls SuperLU, write an algorithm that factorizes the
    # spanning tree conditioner "layer by layer"
    M1 = ilu_to_linear_operator(sparse.linalg.splu(P1))    
    
    # Minimum spanning tree preconditioner
    P2 = ST['min_spanning_tree_adj']
    P2_sc = s_coverage(mtx, P2)
    M2 = ilu_to_linear_operator(sparse.linalg.splu(P2))

    # iLU(0)
    M3 = ilu_to_linear_operator(sparse.linalg.spilu(mtx, drop_tol=0))
    sc = [None, P1_sc, P2_sc, None]
    preconds = [None, M1, M2, M3]
    
    # Use logarithmic scale for relative residual (y-scale)
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('FRE')
    maxiter = 100

    for i, M in enumerate(preconds):
        res = run_trial(mtx, x, M=M, maxiter=maxiter, restart=maxiter)
        label = None

        if i == 0:
            label = 'unpreconditioned'
        if i == 1:
            label = 'maxST'
        if i == 2:
            label = 'minST'
        if i == 3:
            label = 'iLU'

        x_norm = np.linalg.norm(x)
        fre = [np.linalg.norm(x_k - x) / x_norm for x_k in res['x_k']]

        print("{}, {} iters, s_coverage: {}, iters, FRE: {}".format(label, res['iters'], sc[i], fre[-1]))

        # Plot results for specific preconditioner
        print()
        ax.plot(range(1, len(fre)+1), fre, label=label)

    ax.legend(title=f'{title}, x{title_x}')
    fig.savefig(f'{title}_x{title_x}.png')
    plt.close()
    

# %% Real symmetric matrix
mtx1 = mmread('../mtx/c-20.mtx')
np.random.seed(42)
n1, m1 = mtx1.shape

print('mtx1, rhs: normally distributed')
run_trial_precond(mtx1, np.random.randn(n1, 1), title='c-20', title_x='randn')
print()

print('mtx1, rhs: ones')
run_trial_precond(mtx1, np.ones((n1, 1)), title='c-20', title_x='ones')
print()

print('mtx1, rhs: sine')
run_trial_precond(mtx1, np.sin(np.linspace(0, 100*np.pi, n1)), title='c-20', title_x='sine')


# %% Real unsymmetric (100% pattern symmetry, 98.8% numeric symmetry)
mtx2 = mmread('../mtx/ex28.mtx')
np.random.seed(42)
n2, m2 = mtx2.shape

# print('mtx2, rhs: normally distributed')
# run_trial_precond(mtx2, np.random.randn(n2, 1), title='ex28', title_x='randn')
# print()

# print('mtx2, rhs: ones')
# run_trial_precond(mtx2, np.ones((n2, 1)), title='ex28', title_x='ones')
# print()

# print('mtx2, rhs: sine')
# run_trial_precond(mtx2, np.sin(np.linspace(0, 100*np.pi, n2)), title='ex28', title_x='sine')


# %% Real unsymmetric (100% pattern symmetry, 98.8% numeric symmetry), based on
# symmetrization (A + A.T) / 2.
print('mtx2+symm, rhs: normally distributed')
run_trial_precond(mtx2, np.random.randn(n2, 1), symmetrize=True, title='ex28+symm', title_x='randn')
print()

print('mtx2+symm, rhs: ones')
run_trial_precond(mtx2, np.ones((n2, 1)), symmetrize=True, title='ex28+symm', title_x='ones')
print()

print('mtx2+symm, rhs: sine')
run_trial_precond(mtx2, np.sin(np.linspace(0, 100*np.pi, n2)), symmetrize=True, title='ex28+symm', title_x='sine')

