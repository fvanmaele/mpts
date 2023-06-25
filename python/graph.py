#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fvanmaele
"""

# %%
import os
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.io import mmread
from scipy import sparse


# %%
# TODO: return percentage (of numerical, pattern symmetry)
def sparse_is_symmetric(mtx, tol=1e-10):
    """ Check if a matrix is symmetric up to a certain tolerance
    """
    return (abs(mtx-mtx.T) > tol).nnz == 0


def digraph_find_roots(G):
    """ Find roots (no incoming edges) of directed graph.
    """
    assert G.is_directed()
    roots = []

    for deg in G.in_degree:
        if deg[1] == 0:
            roots.append(deg[0])

    return roots, len(roots) == 1


def digraph_maximum_spanning_forest(G):
    """ Find the maximum spanning arboresence for each strongly connected component of a digraph
    """
    assert G.is_directed()
    roots, has_unique_root = digraph_find_roots(G)
    
    if has_unique_root:
        return nx.maximum_spanning_arborescence(G)
    else:
        maximum_spanning_forest = nx.DiGraph()
        strongly_connected_components = nx.strongly_connected_components(G)
        
        for component in strongly_connected_components:
            subgraph = G.subgraph(component)
            maximum_arborescence = nx.maximum_spanning_arborescence(subgraph, attr='weight')
            maximum_spanning_forest = nx.compose(maximum_spanning_forest, maximum_arborescence)
    
        return maximum_spanning_forest


def digraph_minimum_spanning_forest(G):
    """ Find the minimum spanning arboresence for each strongly connected component of a digraph
    """
    assert G.is_directed()
    roots, has_unique_root = digraph_find_roots(G)

    if has_unique_root:
        return nx.minimum_spanning_arborescence(G)
    else:
        minimum_spanning_forest = nx.DiGraph()
        strongly_connected_components = nx.strongly_connected_components(G)
        
        for component in strongly_connected_components:
            subgraph = G.subgraph(component)
            minimum_arborescence = nx.minimum_spanning_arborescence(subgraph, attr='weight')
            minimum_spanning_forest = nx.compose(minimum_spanning_forest, minimum_arborescence)
    
        return minimum_spanning_forest


def sparse_mask(mtx, mtx_mask):
    """ Index a sparse matrix with another sparse matrix of the same dimension.
    
    It is assumed that non-zero indices of `mtx_mask` are a subset of 
    non-zero indices of `mtx` (with potentially differing entries).
    """
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
    """ Compute spanning tree preconditioner for a given sparse matrix.
    """
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
        G_symm = nx.Graph(abs((mtx + mtx.T) / 2))
        maxST_pre = nx.maximum_spanning_tree(G_symm)
        minST_pre = nx.minimum_spanning_tree(G_symm)

    else:
        G_abs = nx.DiGraph(abs(mtx))
        maxST_pre = digraph_maximum_spanning_forest(G_abs)
        minST_pre = digraph_minimum_spanning_forest(G_abs)

    # Construct new sparse matrix based on non-zero weights of spanning tree.
    # The diagonal is added manually, because `nx` does not include weights
    # for self-loops.
    D = sparse.diags(mtx.diagonal())  # DIAgonal
    maxST = sparse_mask(mtx, sparse.coo_array(nx.to_scipy_sparse_array(maxST_pre) + D))
    minST = sparse_mask(mtx, sparse.coo_array(nx.to_scipy_sparse_array(minST_pre) + D))

    return {
        'max_spanning_tree': nx.Graph(maxST) if sparse_is_symmetric(maxST) else nx.DiGraph(maxST), 
        'max_spanning_tree_adj': maxST,
        'min_spanning_tree': nx.Graph(minST) if sparse_is_symmetric(minST) else nx.DiGraph(minST),
        'min_spanning_tree_adj': minST
    }


def s_coverage(mtx, mtx_pruned):
    """ Compute the S-coverage as quality measure for a sparse preconditioner
    """
    assert sparse.issparse(mtx)
    assert sparse.issparse(mtx_pruned)
    
    #sc = sparse.linalg.norm(mtx_pruned, ord=1) / sparse.linalg.norm(mtx, ord=1)
    sc = abs(sparse.csr_matrix(mtx_pruned)).sum() / abs(sparse.csr_matrix(mtx)).sum()
    if sc > 1:
        warnings.warn('S coverage is greater than 1')

    return sc


def lu_sparse_operator(P, inexact=False):
    """ SuperLU based preconditioner for use with GMRES
    """
    assert sparse.issparse(P)
    nrows, ncols = P.shape
    
    # TODO: handle runtime errors (singular preconditioner)
    if inexact:
        M = sparse.linalg.splu(P)
    else:
        M = sparse.linalg.spilu(P, drop_tol=0)  # iLU(0)

    return sparse.linalg.LinearOperator((nrows, ncols), M.solve)


class gmres_counter(object):
    """ Class for counting the number of (inner) GMRES iterations
    """
    def __init__(self):
        self.niter = 0
        self.rk = []  # relative residuals

    def __call__(self, rk=None):
        self.niter += 1
        self.rk.append(rk)


# %%
def run_trial(mtx, x, M=None, maxiter=100, restart=20):
    """ Solve a preconditioned linear system with a fixed number of GMRES iterations
    """
    # Right-hand side from exact solution
    b = mtx * x
    counter = gmres_counter()
    
    # Compare (preconditioned) relative residual in each iteration
    x_gmres, ec = sparse.linalg.gmres(mtx, b, M=M, callback=counter, tol=0, atol=0,
                                      maxiter=maxiter, restart=restart)
    return { 
        'x': x,
        'rk': counter.rk,
        'exit_code': ec, 
        'iters': counter.niter 
    }


def run_trial_precond(mtx, x, maxiter=100, title=None, title_x=None, symmetrize=False, custom=None):
    """ Compare the performance of spanning tree preconditioners
    """
    ST = compute_spanning_trees(mtx, symmetrize=symmetrize)
    
    # Maximum spanning tree preconditioner
    # TODO: handle runtime errors (singular preconditioner)
    P1 = ST['max_spanning_tree_adj']
    P1_sc = s_coverage(mtx, P1)
    M1 = lu_sparse_operator(P1)

    # LU (with the correct permutation) applied to a spanning tree does
    # not result in fill-in.
    # TODO: factorize the spanning tree conditioner "layer by layer

    # Minimum spanning tree preconditioner
    P2 = ST['min_spanning_tree_adj']
    P2_sc = s_coverage(mtx, P2)
    M2 = lu_sparse_operator(P2)

    # iLU(0)
    M3 = lu_sparse_operator(mtx, inexact=True)
    sc = [None, P1_sc, P2_sc, None]
    preconds = [None, M1, M2, M3]

    # Custom preconditioner    
    if custom is not None:
        P4 = mmread(custom)
        M4 = lu_sparse_operator(P4)
        
        preconds.append(M4)
        sc.append(None)

    # Use logarithmic scale for relative residual (y-scale)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.set_dpi(300)
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('relres')

    # Note: preconditioned relres and relres are approximately equal
    for i, M in enumerate(preconds):
        result = run_trial(mtx, x, M=M, maxiter=maxiter, restart=maxiter)
        relres = result['rk']
        label = None

        if i == 0:
            label = 'unpreconditioned'
        if i == 1:
            label = 'maxST'
        if i == 2:
            label = 'minST'
        if i == 3:
            label = 'iLU'
        if i == 4:
            label = 'custom'

        print("{}, {} iters, s_coverage: {}, iters, relres: {}".format(
            label, result['iters'], sc[i], relres[-1]))

        # Plot results for specific preconditioner
        ax.plot(range(1, len(relres)+1), relres, label=label)

    ax.legend(title=f'{title}, x{title_x}')
    fig.savefig(f'{title}_x{title_x}.png')
    plt.close()
    

# %%
def main(mtx_path, seed, restart, maxiter, symmetrize, precond=None):
    np.random.seed(seed)
    mtx   = mmread(mtx_path)
    n, m  = mtx.shape
    title = mtx_path.stem

    # Right-hand sides
    x1 = np.random.randn(n, 1)
    x2 = np.ones((n, 1))
    x3 = np.sin(np.linspace(0, 100*np.pi, n))

    # TODO: handle runtime errors (singular preconditioner)
    print(f'{title}, rhs: normally distributed')
    run_trial_precond(mtx, x1, maxiter=maxiter, title=title, title_x='randn', 
                      symmetrize=symmetrize, custom=precond)
    print(f'\n{title}, rhs: ones')
    run_trial_precond(mtx, x2, maxiter=maxiter, title=title, title_x='ones', 
                      symmetrize=symmetrize, custom=precond)
    print(f'\n{title}, rhs: sine')
    run_trial_precond(mtx, x3, maxiter=maxiter, title=title, title_x='sine', 
                      symmetrize=symmetrize, custom=precond)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='mst_precond', description='trials for spanning tree preconditioner')
    parser.add_argument('--seed',    type=int, default=42, 
                        help='seed for random numbers')
    parser.add_argument('--restart', type=int, default=20, 
                        help='dimension of GMRES subspace')
    parser.add_argument('--maxiter', type=int, default=100, 
                        help='number of GMRES iterations')
    parser.add_argument('--no-symmetrize', dest='symmetrize', action='store_false', 
                        help='use directed spanning tree algorithms instead of symmetrization (slow)')
    parser.add_argument('--precond', type=str, 
                        help='path to preconditioning matrix, to be solved with SuperLU')
    parser.add_argument('mtx', type=str)
    
    args = parser.parse_args()
    main(Path(args.mtx), args.seed, args.restart, args.maxiter, args.symmetrize, args.precond)
