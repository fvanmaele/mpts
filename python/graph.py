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
import ilupp

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


# %% Generalized graph preconditioner
def graph_precond(mtx, optG, symmetrize=False):
    if symmetrize:
        G_abs = nx.Graph(abs((mtx + mtx.T) / 2))
    else:
        G_abs = nx.Graph(abs(mtx))

    D = sparse.diags(mtx.diagonal())  # DIAgonal
    O = optG(G_abs)  # graph optimization function (spanning tree, linear forest, etc.)
    
    return sparse_mask(mtx, sparse.coo_array(nx.to_scipy_sparse_array(O) + D))


def graph_precond_add_m(mtx, optG, m, symmetrize=False, tolist=False):
    # Only consider absolute values for the maximum spanning tree
    if symmetrize:
        R = nx.Graph(abs((mtx + mtx.T) / 2))
    else:
        R = nx.Graph(abs(mtx))

    # Begin with an empty sparse matrix or empty collection thereof
    if tolist:
        S = []
    else:
        S = sparse.csr_matrix(sparse.dok_matrix(mtx.shape))

    # Retrieve diagonal of input matrix for later adding
    D = sparse.diags(mtx.diagonal())

    # In every iteration, the optimized graph is computed for the remainder 
    # (excluding self-loops)
    for k in range(m):
        Mk = nx.to_scipy_sparse_array(optG(R))
        
        # Accumulation of spanning trees (may have any amount of cycles)
        if tolist:
            S.append(Mk)
        else:
            S = S + Mk
        
        # Subtract weights for the next iteration
        R = nx.Graph(nx.to_scipy_sparse_array(R) - Mk)

    if tolist:
        return [sparse_mask(mtx, sparse.coo_array(Sk + D)) for Sk in S]
    else:
        return sparse_mask(mtx, sparse.coo_array(S + D))


# %% Spanning tree preconditioner
def spanning_tree_precond(mtx, symmetrize=False):
    """ Compute spanning tree preconditioner for a given sparse matrix.
    """
    return graph_precond(mtx, nx.maximum_spanning_tree, symmetrize=symmetrize)


def spanning_tree_precond_add_m(mtx, m, symmetrize=False, tolist=False):
    """ Compute m spanning tree factors, computed by subtraction from the original graph.
    """
    return graph_precond_add_m(mtx, nx.maximum_spanning_tree, m, symmetrize=symmetrize, tolist=tolist)


# %% Maximum linear forest preconditioner
def find_n_factors(G, n):
    """Sequential greedy [0,n]-factor computation on a weighted graph G = (V, E)
    """
    # factors = [[] for _ in range(0, mtx.shape[0])]
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


def linear_forest(G):
    assert not G.is_directed()

    Gn = find_n_factors(G, 2)
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

            # Add segment to forest
            forest.update(Gc)

    # Double check that the linear forest is cycle-free
    try:
        nx.find_cycle(forest)
        raise AssertionError
    
    except nx.NetworkXNoCycle:
        return forest
    

def linear_forest_precond(mtx, symmetrize=False):
    """ Compute spanning tree preconditioner for a given sparse matrix.
    """
    return graph_precond(mtx, linear_forest, symmetrize=symmetrize)


def linear_forest_precond_add_m(mtx, m, symmetrize=False, tolist=False):
    """ Compute m spanning tree factors, computed by subtraction from the original graph.
    """
    return graph_precond_add_m(mtx, linear_forest, m, symmetrize=symmetrize, tolist=tolist)


# %%
def lu_sparse_operator(P, inexact=False):
    """ SuperLU based preconditioner for use with GMRES
    """
    assert sparse.issparse(P)
    nrows, ncols = P.shape

    if inexact:
        return ilupp.ILU0Preconditioner(sparse.csc_matrix(P))
    else:
        return sparse.linalg.LinearOperator((nrows, ncols), sparse.linalg.splu(P).solve)


class AltLinearOperator(sparse.linalg.LinearOperator):
    """ Class for non-stationary preconditioners, used with FGMRES.
    """
    def __init__(self, shape, Mi):
        assert len(Mi) > 0

        self.i  = len(Mi)
        self.Mi = Mi
        self.iter = 0       # current iteration, taken modulo `i`
        self.shape = shape  # assumed consistent between `Mi`
        self.dtype = None
        super().__init__(self.dtype, self.shape)

    def _matvec(self, x):
        # Select preconditioner based on current iteration
        Mk = self.Mi[self.iter % self.i]
        self.iter += 1

        # Support objects with `.solve` method and (sparse) matrices
        if callable(Mk):
            v = Mk(x)
        else:
            v = Mk @ x
        return v


class gmres_counter(object):
    """ Class for counting the number of GMRES iterations (inner+outer)
    """
    def __init__(self):
        self.niter = 0
        self.xk = []

    def __call__(self, xk=None):
        self.niter += 1
        self.xk.append(xk)


def run_trial(mtx, x, M, k_max_outer, k_max_inner):
    """ Solve a (right) preconditioned linear system with a fixed number of GMRES iterations
    """
    # Right-hand side from exact solution
    b = mtx * x
    counter = gmres_counter()
    residuals = []  # input vector for fgmres residuals

    try:
        x_gmres, info = krylov.fgmres(mtx, b, M=M, x0=None, tol=1e-15, 
                                      restrt=k_max_inner, maxiter=k_max_outer,
                                      callback=counter, residuals=residuals)

        # Normalize to relative residual
        relres = np.array(residuals) / np.linalg.norm(b)

        # Compute forward relative error
        x_diff = np.matrix(counter.xk) - x.T
        fre = np.linalg.norm(x_diff, axis=1) / np.linalg.norm(x)

        return {
            'x':  x,
            'fre': fre.tolist(),
            'rk': relres,
            'exit_code': info, 
            'iters': counter.niter 
        }

    except ValueError:
        return None


def trial_jacobi(mtx):
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


def trial_tridiag(mtx):
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


def trial_max_st(mtx, mtx_is_symmetric):
    if not mtx_is_symmetric:
        P = spanning_tree_precond(mtx, symmetrize=True)
    else:
        P = spanning_tree_precond(mtx)

    # LU (with the correct permutation) applied to a spanning tree has no fill-in.
    # TODO: factorize the spanning tree conditioner "layer by layer"
    try:
        M = lu_sparse_operator(P)
    except RuntimeError:
        M = None

    return { 
        's_coverage': s_coverage(mtx, P),
        's_degree'  : s_degree(P),
        'precond'   : M
    }


def trial_max_st_add_m(mtx, mtx_is_symmetric, m):
    assert m > 1
    
    if not mtx_is_symmetric:
        P = spanning_tree_precond_add_m(mtx, m, symmetrize=True)
    else:
        P = spanning_tree_precond_add_m(mtx, m)

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


def trial_max_lf(mtx, mtx_is_symmetric):
    if not mtx_is_symmetric:
        P = linear_forest_precond(mtx, symmetrize=True)
    else:
        P = linear_forest_precond(mtx)

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


def trial_max_lf_add_m(mtx, mtx_is_symmetric, m):
    assert m > 1
    
    if not mtx_is_symmetric:
        P = linear_forest_precond_add_m(mtx, m, symmetrize=True)
    else:
        P = linear_forest_precond_add_m(mtx, m)

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


def trial_ilu0(mtx):
    try:
        M = lu_sparse_operator(mtx, inexact=True)
    except RuntimeError:
        M = None
    
    return {
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


def trial_custom(mtx, P):
    try:
        M = lu_sparse_operator(P)
    except RuntimeError:
        M = None

    return {
        's_coverage': None,
        's_degree'  : s_degree(P),
        'precond'   : M
    }


def run_trial_precond(mtx, x, k_max_outer=10, k_max_inner=20, title=None, title_x=None, custom=None):
    """ Compare the performance of spanning tree preconditioners
    """
    mtx_is_symmetric = sparse_is_symmetric(mtx)
    sc = []
    sd = []
    labels = []
    preconds = []

    # # Unpreconditioned system
    # sc.append(1)
    # sd.append(s_degree(mtx))
    # preconds.append(None)
    # labels.append('unpreconditioned')    

    # # Jacobi preconditioner
    # jacobi = trial_jacobi(mtx)
    # sc.append(jacobi['s_coverage'])
    # sd.append(jacobi['s_degree'])
    # preconds.append(jacobi['precond'])
    # labels.append('jacobi')

    # # Tridiagonal preconditioner
    # tridiag = trial_tridiag(mtx)
    # sc.append(tridiag['s_coverage'])
    # sd.append(tridiag['s_degree'])
    # preconds.append(tridiag['precond'])
    # labels.append('tridiag')

    # Maximum spanning tree preconditioner
    max_st = trial_max_st(mtx, mtx_is_symmetric)
    sc.append(max_st['s_coverage'])
    sd.append(max_st['s_degree'])
    preconds.append(max_st['precond'])
    labels.append('maxST')
    
    # # Maximum spanning tree preconditioner, additive factors (m = 2..5)
    # for m in range(2, 6):
    #     max_st_add_m = trial_max_st_add_m(mtx, mtx_is_symmetric, m)
    #     sc.append(max_st_add_m['s_coverage'])
    #     sd.append(max_st_add_m['s_degree'])
    #     preconds.append(max_st_add_m['precond'])
    #     labels.append(f'maxST+ (m = {m})')
    
    # Maximum linear forest preconditioner
    # max_lf = trial_max_lf(mtx, mtx_is_symmetric)
    # sc.append(max_lf['s_coverage'])
    # sd.append(max_lf['s_degree'])
    # preconds.append(max_lf['precond'])
    # labels.append('maxLF')
    
    # Maximum linear forest preconditioner, additive factors (m = 2..5)
    # for m in range(2, 6):
    #     max_lf_add_m = trial_max_lf_add_m(mtx, mtx_is_symmetric, m)
    #     sc.append(max_lf_add_m['s_coverage'])
    #     sd.append(max_lf_add_m['s_degree'])
    #     preconds.append(max_lf_add_m['precond'])
    #     labels.append(f'maxLF+ (m = {m})')

    # iLU(0)
    ilu0 = trial_ilu0(mtx)
    sc.append(None)
    sd.append(None)
    preconds.append(ilu0['precond'])
    labels.append('iLU')

    # Custom preconditioner    
    if custom is not None:
        trial_custom(mtx, mmread(custom))
        sc.append(None)
        sd.append(trial_custom['s_degree'])
        labels.append('custom')


    # Use logarithmic scale for relative residual (y-scale)
    fig1, ax1 = plt.subplots()    
    fig1.set_size_inches(8, 6)
    fig1.set_dpi(300)

    ax1.set_yscale('log')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('relres')

    # TODO: subplot for relres (left) and forward relative error (right)
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(8, 6)
    fig2.set_dpi(300)
    
    ax2.set_yscale('log')
    ax2.set_xlabel('iterations')
    ax2.set_ylabel('fre')

    for i, label in enumerate(labels):
        M = preconds[i]

        if M is None and i > 0:
            print("{}, s_coverage: {}, s_degree: {}".format(label, sc[i], sd[i]))
            continue

        result = run_trial(mtx, x, M=M, k_max_outer=k_max_outer, k_max_inner=k_max_inner)

        if result is not None:
            relres = result['rk']
            fre    = result['fre']
    
            print("{}, {} iters, s_coverage: {}, s_degree: {}, relres: {}, fre: {}".format(
                label, result['iters'], sc[i], sd[i], relres[-1], fre[-1]))
    
            # Plot results for specific preconditioner
            ax1.plot(range(1, len(relres)+1), relres, label=label)
            ax2.plot(range(1, len(fre)+1), fre, label=label)

        else:
            warnings.warn(f'failed to solve {label} system')

    ax1.legend(title=f'{title}, x{title_x}')
    fig1.savefig(f'{title}_x{title_x}.png')
    
    ax2.legend(title=f'{title}, x{title_x}')
    fig2.savefig(f'{title}_x{title_x}_fre.png')

    plt.close()
    

# %%
def main(mtx_path, seed, max_outer, max_inner, precond=None):
    np.seterr(all='raise')
    
    np.random.seed(seed)
    mtx   = mmread(mtx_path)
    n, m  = mtx.shape
    title = mtx_path.stem

    # Remove explicit zeros set in some matrices (in-place)
    mtx.eliminate_zeros()

    # Right-hand sides
    x1 = np.random.randn(n, 1)
    x2 = np.ones((n, 1))
    x3 = np.sin(np.linspace(0, 100*np.pi, n))

    print(f'{title}, rhs: normally distributed')
    run_trial_precond(mtx, x1, k_max_outer=max_outer, k_max_inner=max_inner, 
                      title=title, title_x='randn', custom=precond)
    
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
