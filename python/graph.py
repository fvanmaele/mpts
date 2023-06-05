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

def cond(A):
    if sparse.issparse(A):
        raise NotImplementedError()
    else:
        return np.format_float_scientific(np.linalg.cond(A))

# Check if a matrix is symmetric up to a certain tolerance
def sparse_is_symmetric(mtx, tol=1e-10):
    return (abs(mtx-mtx.T) > tol).nnz == 0

# Symmetrize a matrix using an l1 criterion
def symmetrize_l1(mtx, factor=1):
    assert factor >= 1
    assert sparse.issparse(mtx)
    mtx_csr = sparse.csr_matrix(mtx)

    rows = []
    cols = []
    data = []
    done = {}

    for i, j, v in zip(mtx.row, mtx.col, mtx.data):
        if (i, j) in done:
            continue
        v_max = None

        if abs(mtx_csr[i, j]) > abs(factor*mtx_csr[j, i]):
            v_max = mtx_csr[i, j]
        else:
            v_max = mtx_csr[j, i]

        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([v_max, v_max])
        
        done[(i, j)] = 1
        done[(j, i)] = 1

    return sparse.coo_matrix((data, (rows, cols)))


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


# TODO: add diagnostics
def compute_spanning_trees(mtx, symmetrize=False):
    assert sparse.issparse(mtx)
    G, maxST, minST = None, None, None

    if sparse_is_symmetric(mtx):
        G = nx.Graph(mtx)
        maxST = nx.maximum_spanning_tree(G)
        minST = nx.minimum_spanning_tree(G)
    
    elif symmetrize:
        G = nx.DiGraph(mtx)
        G_symm = nx.Graph(symmetrize_l1(mtx))
        maxST_symm = nx.maximum_spanning_tree(G_symm)
        minST_symm = nx.minimum_spanning_tree(G_symm)
        
        # Find the intersection between G and G_symm, preserving weights in G
        # XXX: use sparse matrices
        maxST_mask = nx.to_numpy_array(maxST_symm) != 0
        minST_mask = nx.to_numpy_array(minST_symm) != 0

        # XXX: s-coverage is not between 0 and 1 for minST        
        maxST = nx.Graph(np.where(maxST_mask, mtx.todense(), 0))
        minST = nx.Graph(np.where(minST_mask, mtx.todense(), 0))

    else:
        G = nx.DiGraph(mtx)
        roots, has_unique_root = digraph_find_roots(G)

        if has_unique_root:
            maxST = nx.maximum_spanning_arborescence(G)
            minST = nx.minimum_spanning_arborescence(G)
        else:
            maxST = digraph_maximum_spanning_forest(G)
            minST = digraph_minimum_spanning_forest(G)
    
    # TODO: assert that diagonal of preconditioner and of matrix are identical
    D = sparse.diags(mtx.diagonal())     # DIAgonal

    # XXX: networkx does not include weights for self-loops (diagonal elements in the adjacency graph)
    return {
        'graph': G,
        'max_spanning_tree': maxST, 
        'max_spanning_tree_adj': sparse.coo_matrix(nx.to_scipy_sparse_array(maxST) + D),
        'min_spanning_tree': minST,
        'min_spanning_tree_adj': sparse.coo_matrix(nx.to_scipy_sparse_array(minST) + D)
    }


# Compute the S-coverage as quality measure for a sparse preconditioner
def s_coverage(mtx, mtx_pruned):
    assert sparse.issparse(mtx)
    assert sparse.issparse(mtx_pruned)
    
    sc = sparse.linalg.norm(mtx_pruned, ord=1) / sparse.linalg.norm(mtx, ord=1)
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
    
    # Perform a tixed number of iterations
    float_min = sys.float_info.min
    tol  = float_min
    atol = float_min
    
    # Keep track of GMRES (inner) iterations
    counter = gmres_counter()
    
    # callback_type='x' to compare FRE in each iteration
    x_gmres, ec = sparse.linalg.gmres(mtx, b, M=M, callback=counter, callback_type='x', 
                                      maxiter=maxiter, restart=restart, tol=tol, atol=atol)
    return { 'x': x, 'x_k': counter.xk, 'exit_code': ec, 'iters': counter.niter }


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
    # TODO: S-coverage?
    M3 = ilu_to_linear_operator(sparse.linalg.spilu(mtx, drop_tol=0))
    sc = [None, P1_sc, P2_sc, None]

    # Use logarithmic scale for relative residual (y-scale)    
    #plt.yscale('log')

    for i, M in enumerate([None, M1, M2, M3]):
        res = run_trial(mtx, x, M=M)
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
        #fre = [np.linalg.norm(xk - x) / x_norm for xk in res['x_k']]
        x_gmres = res['x_k'][-1]
        fre = np.linalg.norm(x_gmres - x) / x_norm

        print("{}, {} iters, s_coverage: {}, iters, FRE: {}".format(
            label, res['iters'], sc[i], fre))

        # Plot results for specific preconditioner
        # plt.plot(res['iters'], res['relres'], label=label)
        # plt.legend(title=f'{title}, x{title_x}')


# %% Real symmetric, positive definite (TODO: diagonally dominant?)
mtx1 = mmread('../mtx/ufsmc/494_bus.mtx')
np.random.seed(42)
n1, m1 = mtx1.shape

print('mtx1, rhs: normally distributed')
run_trial_precond(mtx1, np.random.randn(n1, 1))
print()

print('mtx1, rhs: ones')
run_trial_precond(mtx1, np.ones((n1, 1)))
print()

print('mtx1, rhs: sine')
run_trial_precond(mtx1, np.sin(np.linspace(0, 100*np.pi, n1)))


# %% Real unsymmetric (76% non-zero pattern symmetry), non positive definite
mtx2 = mmread('../mtx/ufsmc/arc130.mtx')
np.random.seed(42)
n2, m2 = mtx2.shape

print('mtx2, rhs: normally distributed')
run_trial_precond(mtx2, np.random.randn(n2, 1))
print()

print('mtx2, rhs: ones')
run_trial_precond(mtx2, np.ones((n2, 1)))
print()

print('mtx2, rhs: sine')
run_trial_precond(mtx2, np.sin(np.linspace(0, 100*np.pi, n2)))


# %% Spanning tree preconditioner based on an l1 symmetrization of the original matrix.
print('mtx2+symm, rhs: normally distributed')
run_trial_precond(mtx2, np.random.randn(n2, 1), symmetrize=True)
print()

print('mtx2+symm, rhs: ones')
run_trial_precond(mtx2, np.ones((n2, 1)), symmetrize=True)
print()

print('mtx2+symm, rhs: sine')
run_trial_precond(mtx2, np.sin(np.linspace(0, 100*np.pi, n2)), symmetrize=True)
