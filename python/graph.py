#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fvanmaele
"""

# %%
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from scipy.io import mmread
from scipy import sparse
from pyamg import krylov

# %% Sparse matrix helpers
def sparse_is_symmetric(mtx, tol=1e-10):
    """ Check a matrix for numerical symmetry
    """
    # XXX: does networkx use any tolerances when constructing an (undirected) graph from a matrix?
    return (abs(mtx-mtx.T) > tol).nnz == 0


def sparse_mask(mtx, mtx_mask):
    """ Index a sparse matrix with another sparse matrix of the same dimension.
    
    It is assumed that non-zero indices of `mtx_mask` are a subset of 
    non-zero indices of `mtx` (with potentially differing entries).
    """
    # TODO: do a precondition check on "is a subset of"
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


def lu_sparse_operator(P, inexact=False):
    """ SuperLU based preconditioner for use with GMRES
    """
    assert sparse.issparse(P)
    nrows, ncols = P.shape

    if inexact:
        M = sparse.linalg.splu(P)
    else:
        M = sparse.linalg.spilu(P, drop_tol=0)  # iLU(0)

    return sparse.linalg.LinearOperator((nrows, ncols), M.solve)


class gmres_counter(object):
    """ Class for counting the number of GMRES iterations (inner+outer)
    """
    def __init__(self):
        self.niter = 0
        self.xk = []  # relative residuals

    def __call__(self, xk=None):
        self.niter += 1
        self.xk.append(xk)


def s_coverage(mtx, mtx_pruned):
    """ Compute the S-coverage as quality measure for a sparse preconditioner
    """
    assert sparse.issparse(mtx)
    assert sparse.issparse(mtx_pruned)
    
    sc = abs(sparse.csr_matrix(mtx_pruned)).sum() / abs(sparse.csr_matrix(mtx)).sum()
    if sc > 1:
        warnings.warn('S coverage is greater than 1')

    return sc


def s_degree(mtx):
    """Compute the maximum degree for the graph of an adjacency matrix.
    
    Self-loops (diagonal elements) are ignored.
    """
    assert sparse.issparse(mtx)
    
    M = sparse.csr_matrix(mtx - sparse.diags(mtx.diagonal()))
    M_deg = [M.getrow(i).getnnz() for i in range(M.shape[0])]

    return max(M_deg)


# %% Spanning tree preconditioner
# TODO: allow setting maximum degree (use find_n_factors with n)
def spanning_tree_precond(mtx, symmetrize=False):
    """ Compute spanning tree preconditioner for a given sparse matrix.
    """
    # Note: networkx uses the actual weights, and not their absolute values, to 
    # determine the minimum/maximum spanning tree.
    if symmetrize:
        G_abs = nx.Graph(abs((mtx + mtx.T) / 2))
    else:
        G_abs = nx.Graph(abs(mtx))

    # Construct new sparse matrix based on non-zero weights of spanning tree.
    # The diagonal is added manually, because `nx` does not include weights
    # for self-loops.
    D = sparse.diags(mtx.diagonal())  # DIAgonal
    maxST_pre = nx.maximum_spanning_tree(G_abs)  # maximum spanning forest (if not connected)
    
    return sparse_mask(mtx, sparse.coo_array(nx.to_scipy_sparse_array(maxST_pre) + D))


# %% Maximum linear forest preconditioner
def find_n_factors(mtx, n):
    """Sequential greedy [0,n]-factor computation on a weighted graph G = (V, E)
    """
    assert sparse_is_symmetric(mtx)

    # factors = [[] for _ in range(0, mtx.shape[0])]
    G = nx.Graph(mtx)  # absolute values for edge weights
    G.remove_edges_from(nx.selfloop_edges(G))  # ignore self-loops (diagonal elements)

    Gn = nx.Graph()
    Gn.add_nodes_from(G)

    # Iterate over edges in decreasing (absolute) weight order
    for v, w, O in sorted(G.edges(data=True), key=lambda t: abs(t[2].get('weight', 1)), reverse=True):
        assert(v != w)  # sanity check

        # Empty set, one vertex, ..., or n vertices from the neighborhood of a vertex v
        if Gn.degree[v] < n and Gn.degree[w] < n:
            Gn.add_edge(v, w, weight=O['weight'])

        # if len(factors[v]) < n and len(factors[w]) < n and v != w:
        #     factors[v].append(w)  # undirected graph
        #     factors[w].append(v)

    return Gn  # does not include loops


def linear_forest(mtx):
    assert sparse_is_symmetric(mtx)

    Gn = find_n_factors(mtx, 2)
    forest = nx.Graph()
    forest.add_nodes_from(Gn)

    for c in nx.connected_components(Gn):
        Gc = Gn.subgraph(c).copy()
        Cb = nx.cycle_basis(Gc)
        assert len(Cb) < 2

        if len(Cb) == 0:
            # OK, add to forest
            forest.update(Gc)
        else:
            # Break cycle by removing edge of smallest (absolute) weight
            e_min = min(Gc.edges(data=True), key=lambda t: abs(t[2].get('weight', 1)))
            Gc.remove_edge(e_min[0], e_min[1])

            forest.update(Gc)

    # Double check that the linear forest is cycle-free
    try:
        nx.find_cycle(forest)
        raise AssertionError
    except nx.NetworkXNoCycle:
        return forest
    

def linear_forest_precond(mtx, symmetrize=False):
    # TODO: common logic with spanning_tree_precond()
    if symmetrize:
        LF = linear_forest(abs((mtx + mtx.T) / 2))
    else:
        LF = linear_forest(abs(mtx))  # Note: abs() already done in linear_forest()
    
    D = sparse.diags(mtx.diagonal())  # DIAgonal

    return sparse_mask(mtx, sparse.coo_array(nx.to_scipy_sparse_array(LF) + D))


# %%
def run_trial(mtx, x, M, k_max_outer, k_max_inner):
    """ Solve a (right) preconditioned linear system with a fixed number of GMRES iterations
    """
    # Right-hand side from exact solution
    b = mtx * x
    counter = gmres_counter()
    residuals = []

    # Compare relative residual in each iteration
    x_gmres, info = krylov.fgmres(mtx, b, M=M, x0=None, tol=0, restrt=k_max_inner, maxiter=k_max_outer,
                                  callback=counter, residuals=residuals)
    relres = np.array(residuals) / np.linalg.norm(b)

    return { 
        'x':  x,
        'xk': counter.xk,
        'rk': relres,
        'exit_code': info, 
        'iters': counter.niter 
    }


def run_trial_precond(mtx, x, k_max_outer=10, k_max_inner=20, title=None, title_x=None, custom=None):
    """ Compare the performance of spanning tree preconditioners
    """
    mtx_is_symmetric = sparse_is_symmetric(mtx)

    # Unpreconditioned system
    sc = [1]
    sd = [s_degree(mtx)]
    preconds = [None]
    labels = ['unpreconditioned']

    
    # Jacobi preconditioner
    PJ = mtx.diagonal()

    sc.append(s_coverage(mtx, sparse.diags(PJ)))
    sd.append(s_degree(sparse.diags(PJ)))
    labels.append('jacobi')

    try:
        preconds.append(sparse.diags(1. / PJ, format='csc'))
    except FloatingPointError:
        preconds.append(None)


    # Tridiagonal preconditioner
    diagonals = [mtx.diagonal(k=-1), mtx.diagonal(k=0), mtx.diagonal(k=1)]
    PT = sparse.diags(diagonals, [-1, 0, 1])
    
    sc.append(s_coverage(mtx, PT))
    sd.append(s_degree(PT))
    labels.append('tridiag')

    try:
        preconds.append(lu_sparse_operator(PT))
    except RuntimeError:
        preconds.append(None)


    # 1) Maximum spanning tree preconditioner
    # LU (with the correct permutation) applied to a spanning tree has no fill-in.
    # TODO: factorize the spanning tree conditioner "layer by layer"
    if not mtx_is_symmetric:
        ST = spanning_tree_precond(mtx, symmetrize=True)
    else:
        ST = spanning_tree_precond(mtx)
    
    sc.append(s_coverage(mtx, ST))
    sd.append(s_degree(ST))
    labels.append('maxST')

    try:
        preconds.append(lu_sparse_operator(ST))
    except RuntimeError:
        preconds.append(None)

    
    # 2) Maximum linear forest preconditioner
    # Note: not permuted to tridiagonal system
    if not mtx_is_symmetric:
        LF = linear_forest_precond(mtx, symmetrize=True)
    else:
        LF = linear_forest_precond(mtx)

    sc.append(s_coverage(mtx, LF))
    sd.append(s_degree(LF))
    labels.append('maxLF')

    try:
        preconds.append(lu_sparse_operator(LF))    
    except RuntimeError:
        preconds.append(None)
    

    # 3) iLU(0)
    sc.append(None)
    sd.append(None)
    labels.append('iLU')

    try:
        preconds.append(lu_sparse_operator(mtx, inexact=True))
    except RuntimeError:
        preconds.append(None)


    # 4) Custom preconditioner    
    if custom is not None:
        PC = mmread(custom)

        sc.append(None)
        sd.append(s_degree(PC))
        labels.append('custom')
        
        try:
            preconds.append(lu_sparse_operator(PC))    
        except RuntimeError:
            preconds.append(None)


    # Use logarithmic scale for relative residual (y-scale)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.set_dpi(300)
    
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('relres')

    for i, label in enumerate(labels):
        M = preconds[i]

        if M is None and i > 0:
            print("{}, s_coverage: {}, s_degree: {}".format(label, sc[i], sd[i]))
            continue

        result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)
        relres = result['rk']
        sols   = result['xk']
        x_norm = np.linalg.norm(x)
        fre    = [np.linalg.norm(xk - x) / x_norm for xk in sols]

        print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
            label, result['iters'], sc[i], sd[i], relres[-1], fre[-1]))

        # Plot results for specific preconditioner
        # TODO: subplot for relres (left) and forward relative error (right)
        ax.plot(range(1, len(relres)+1), relres, label=label)

    ax.legend(title=f'{title}, x{title_x}')
    fig.savefig(f'{title}_x{title_x}.png')
    plt.close()
    

# %%
def main(mtx_path, seed, max_outer, max_inner, precond=None):
    np.seterr(all='raise')
    
    np.random.seed(seed)
    mtx   = mmread(mtx_path)
    n, m  = mtx.shape
    title = mtx_path.stem

    # Right-hand sides
    x1 = np.random.randn(n, 1)
    x2 = np.ones((n, 1))
    x3 = np.sin(np.linspace(0, 100*np.pi, n))

    print(f'{title}, rhs: normally distributed')
    run_trial_precond(mtx, x1, k_max_outer=max_outer, k_max_inner=max_inner, 
                      title_x='randn', custom=precond)
    
    print(f'\n{title}, rhs: ones')
    run_trial_precond(mtx, x2, k_max_outer=max_outer, k_max_inner=max_inner, 
                      title=title, title_x='ones', custom=precond)
    
    print(f'\n{title}, rhs: sine')
    run_trial_precond(mtx, x3, k_max_outer=max_outer, k_max_inner=max_inner, 
                      title=title, title_x='sine', custom=precond)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='mst_precond', description='trials for spanning tree preconditioner')
    parser.add_argument('--seed', type=int, default=42, 
                        help='seed for random numbers')
    parser.add_argument('--max-outer', type=int, default=15, 
                        help='maximum number of outer GMRES iterations')
    parser.add_argument('--max-inner', type=int, default=20,
                        help='maximum number of inner GMRES iterations')
    parser.add_argument('--precond', type=str, 
                        help='path to preconditioning matrix, to be solved with SuperLU')
    parser.add_argument('mtx', type=str)
    
    args = parser.parse_args()
    main(Path(args.mtx), args.seed, args.max_outer, args.max_inner, args.precond)
