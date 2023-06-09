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


def sparse_mask(mtx, mtx_mask, scale=None):
    """ Index a sparse matrix with another sparse matrix of the same dimension.
    
    It is assumed that non-zero indices of `mtx_mask` are a subset of 
    non-zero indices of `mtx` (with potentially differing entries).

    If the `scale` parameter is specified, indexed entries are scaled with a given
    factor instead of removed.
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
        if scale is not None:
            rows.append(i)
            cols.append(j)
            
            if (i, j) in mask:
                data.append(scale*v)
            else:
                data.append(v)
        else:
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


def prune_sparse_matrix(matrix, N):
    #assert sparse.isspmatrix_csr(matrix)
    # Find the nonzero elements and their corresponding row indices
    row_indices, col_indices, values = sparse.find(matrix)

    # Iterate over the unique row indices
    unique_row_indices = np.unique(row_indices)

    for row_idx in unique_row_indices:
        # Find the indices of nonzero elements in the current row
        row_indices_mask = (row_indices == row_idx)
        row_values = values[row_indices_mask]
        row_col_indices = col_indices[row_indices_mask]

        if len(row_values) <= N:
            continue

        # Sort the row values by their absolute values
        sorted_indices = np.argsort(np.abs(row_values))

        # Prune the row values and indices beyond N
        pruned_indices = row_col_indices[sorted_indices[:-N]]
        #pruned_values = row_values[sorted_indices[:-N]]

        # Set the pruned values and indices in the matrix
        matrix[row_idx, pruned_indices] = 0
        #matrix[row_idx, pruned_indices] = pruned_values

    return matrix.tocoo()


# %% Generalized graph preconditioner
# In every case, `optG` is assumed to be a graph optimization function that does
# NOT return self-loops (diagonal elements in the adjacency matrix)
def graph_precond(mtx, optG, symmetrize=False):
    if symmetrize:
        G_abs = nx.Graph(abs((mtx + mtx.T) / 2))
    else:
        G_abs = nx.Graph(abs(mtx))

    D = sparse.diags(mtx.diagonal())  # DIAgonal
    O = optG(G_abs)  # graph optimization function (spanning tree, linear forest, etc.)
    
    return sparse_mask(mtx, sparse.coo_array(nx.to_scipy_sparse_array(O) + D))


def graph_precond_add_m(mtx, optG, m, symmetrize=False):
    # Only consider absolute values for the maximum spanning tree
    if symmetrize:
        R = nx.Graph(abs((mtx + mtx.T) / 2))
    else:
        R = nx.Graph(abs(mtx))

    # Begin with an empty sparse matrix
    S = sparse.csr_matrix(sparse.dok_matrix(mtx.shape))

    # Retrieve diagonal of input matrix for later adding
    D = sparse.diags(mtx.diagonal())

    # In every iteration, the optimized graph is computed for the remainder 
    # (excluding self-loops)
    for k in range(m-1):
        Mk = nx.to_scipy_sparse_array(optG(R))  # no diagonal elements
        
        # Accumulation of spanning trees (may have any amount of cycles)
        S = S + Mk
        
        # Subtract weights for the next iteration
        R = nx.Graph(nx.to_scipy_sparse_array(R) - Mk)

    return sparse_mask(mtx, sparse.coo_array(S + D))


# TODO: pruning
def graph_precond_list_m(mtx, optG, m, symmetrize=False, scale=None, prune=None):
    if symmetrize:
        R = nx.Graph(abs((mtx + mtx.T) / 2))
    else:
        R = nx.Graph(abs(mtx))

    M = nx.to_scipy_sparse_array(optG(nx.Graph(R)))  # no diagonal elements
    S = [M]

    # Retrieve diagonal of input matrix for later adding
    D = sparse.diags(mtx.diagonal())
    
    # Apply graph optimization to graph with entries scaled according to indices
    # retrieved in the previous step
    for k in range(1, m):
        R = nx.Graph(sparse_mask(sparse.coo_array(nx.to_scipy_sparse_array(R)), 
                     sparse.coo_array(M + D), scale=scale))
        M = nx.to_scipy_sparse_array(optG(R))       
        S.append(M)

    return [sparse_mask(mtx, sparse.coo_array(Sk + D)) for Sk in S]


# TODO: check implementation, diagonal factor T
def graph_precond_inv_m(mtx, m, optG, symmetrize=False, prune=False):
    assert sparse.issparse(mtx)
    assert callable(optG)

    if symmetrize:
        R = sparse.coo_array(abs((mtx + mtx.T) / 2))
    else:
        R = sparse.coo_array(abs(mtx))

    D = sparse.diags(mtx.diagonal())
    P = nx.to_scipy_sparse_array(optG(nx.Graph(R)))
    M = sparse_mask(mtx, sparse.coo_array(P + D), scale=None)
    B = mtx.copy()

    for k in range(1, m):
        P = nx.to_scipy_sparse_array(optG(nx.Graph(B @ sparse.linalg.inv(M))))
        B = M.copy()
        M = sparse_mask(mtx, sparse.coo_array(P + D), scale=None)
    
    return M


# %% Spanning tree preconditioner
def spanning_tree_precond(mtx, symmetrize=False):
    """ Compute spanning tree preconditioner for a given sparse matrix.
    """
    return graph_precond(mtx, nx.maximum_spanning_tree, symmetrize=symmetrize)


def spanning_tree_precond_add_m(mtx, m, symmetrize=False):
    """ Compute m spanning tree factors, computed by subtraction from the original graph.
    """
    return graph_precond_add_m(mtx, nx.maximum_spanning_tree, m, symmetrize=symmetrize)


def spanning_tree_precond_list_m(mtx, m, symmetrize=False, scale=None):
    """ Compute m spanning tree factors, computed by successively scaling of optimal entries
    """
    return graph_precond_list_m(mtx, nx.maximum_spanning_tree, m, symmetrize=symmetrize, scale=scale)


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
    """ Compute linear forest preconditioner for a given sparse matrix.
    """
    return graph_precond(mtx, linear_forest, symmetrize=symmetrize)


def linear_forest_precond_add_m(mtx, m, symmetrize=False):
    """ Compute m linear forest factors, computed by subtraction from the original graph.
    """
    return graph_precond_add_m(mtx, linear_forest, m, symmetrize=symmetrize)


def linear_forest_precond_list_m(mtx, m, symmetrize=False, scale=0):
    """ Compute m linear forest factors, computed by successively scaling of optimal entries
    """
    return graph_precond_list_m(mtx, linear_forest, m, symmetrize=symmetrize, scale=scale)


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

        # LinearOperator
        self.shape = shape  # assumed consistent between `Mi`
        self.dtype = None        
        super().__init__(self.dtype, self.shape)
        
        # Child members
        self.i = len(Mi)
        self.Mi = Mi
        self.iter = 0       # current iteration, taken modulo `i`

    def _matvec(self, x):
        # Select preconditioner (inverse) based on current iteration
        M = self.Mi[self.iter % self.i]
        self.iter += 1

        # Support objects with `.solve` method and (sparse) matrices
        return M(x) if callable(M) else M @ x


class IterLinearOperator(sparse.linalg.LinearOperator):
    """ Class for iterative refinement with a series of preconditioners
    """
    def __init__(self, A, Mi, repeat_i=0):
        assert len(Mi) > 0
        assert sparse.issparse(A)

        # Iterative refinement with single preconditioner
        if repeat_i > 0:
            assert len(Mi) == 1

        # LinearOperator
        self.shape = A.shape
        self.dtype = None
        super().__init__(self.dtype, self.shape)

        # Child members
        self.A = A
        if repeat_i > 0:
            self.Mi = Mi * repeat_i
        else:
            self.Mi = Mi

    def _matvec(self, x):
        # Initial estimate
        xk = np.zeros_like(x)

        # Loop over series of preconditioners
        for M in self.Mi:
            if callable(M):
                xk += M(x - self.A @ xk)
            else:
                xk += M @ (x - self.A @ xk)

        return xk


# %%
def solve_rtl(x, solve_l):
    assert len(solve_l) > 1
    v = solve_l[0](x)
    
    for i in range(1, len(solve_l)):
        v = solve_l[i](v)
    return v


def solve_ltr(x, solve_l):
    assert len(solve_l) > 1
    v = solve_l[-1](x)
    
    for i in reversed(range(0, len(solve_l)-1)):
        v = solve_l[i](v)
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
    rhs = mtx * x
    counter = gmres_counter()
    residuals = []  # input vector for fgmres residuals

    try:
        x_gmres, info = krylov.fgmres(mtx, rhs, M=M, x0=None, tol=1e-15, 
                                      restrt=k_max_inner, maxiter=k_max_outer,
                                      callback=counter, residuals=residuals)

        # Normalize to relative residual
        relres = np.array(residuals) / np.linalg.norm(rhs)

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


# %%
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


def trial_max_st(mtx, mtx_is_symmetric, prune=None):
    if not mtx_is_symmetric:
        P = spanning_tree_precond(mtx, symmetrize=True)
    else:
        P = spanning_tree_precond(mtx)

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


# TODO: inverse of each factor M_i (right multiplication)
def trial_max_st_mult_m(mtx, mtx_is_symmetric, m, scale):
    assert m > 1

    if not mtx_is_symmetric:
        Pi = spanning_tree_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = spanning_tree_precond_list_m(mtx, m, scale=scale)
    
    try:
        # Create sparse operator which applies Mi successively
        f = lambda x: solve_rtl(x, [sparse.linalg.splu(P).solve for P in Pi])
        M = sparse.linalg.LinearOperator(mtx.shape, f)

    except RuntimeError:
        M = None
        
    return { 
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


def trial_max_st_alt_i(mtx, mtx_is_symmetric, m, scale):
    assert m > 1

    if not mtx_is_symmetric:
        Pi = spanning_tree_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = spanning_tree_precond_list_m(mtx, m, scale=scale)

    try:
        M = AltLinearOperator(mtx.shape, [sparse.linalg.splu(P).solve for P in Pi])

    except RuntimeError:
        M = None

    return {
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


def trial_max_st_alt_o(mtx, mtx_is_symmetric, m, scale, repeat_i=0):
    if m == 1:
        assert repeat_i > 0

    if not mtx_is_symmetric:
        Pi = spanning_tree_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = spanning_tree_precond_list_m(mtx, m, scale=scale)

    try:
        M = IterLinearOperator(mtx, [sparse.linalg.splu(P).solve for P in Pi], repeat_i=repeat_i)

    except RuntimeError:
        M = None

    return {
            's_coverage': None,
            's_degree'  : None,
            'precond'   : M
        }


def trial_max_st_inv_m(mtx, mtx_is_symmetric, m, prune=None):
    symmetrize = not mtx_is_symmetric
        
    try:
        P = graph_precond_inv_m(mtx, m, nx.maximum_spanning_tree, symmetrize=symmetrize)
    except RuntimeError:
        P = None

    if P is not None:
        try:
            M = lu_sparse_operator(P)
        except RuntimeError:
            M = None

    return { 
        's_coverage': s_coverage(mtx, P) if P is not None else None,
        's_degree'  : s_degree(P) if P is not None else None,
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


def trial_max_lf_mult_m(mtx, mtx_is_symmetric, m, scale):
    assert m > 1

    if not mtx_is_symmetric:
        Pi = linear_forest_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = linear_forest_precond_list_m(mtx, m, scale=scale)
    
    try:
        # Create sparse operator which applies Mi successively
        f = lambda x: solve_rtl(x, [sparse.linalg.splu(P).solve for P in Pi])
        M = sparse.linalg.LinearOperator(mtx.shape, f)

    except RuntimeError:
        M = None
        
    return { 
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


def trial_max_lf_alt_i(mtx, mtx_is_symmetric, m, scale):
    assert m > 1

    if not mtx_is_symmetric:
        Pi = linear_forest_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = linear_forest_precond_list_m(mtx, m, scale=scale)
    
    try:
        M = AltLinearOperator(mtx.shape, [sparse.linalg.splu(P).solve for P in Pi])

    except RuntimeError:
        M = None

    return {
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


def trial_max_lf_alt_o(mtx, mtx_is_symmetric, m, scale, repeat_i=0):
    if m == 1:
        assert repeat_i > 0

    if not mtx_is_symmetric:
        Pi = linear_forest_precond_list_m(mtx, m, symmetrize=True, scale=scale)
    else:
        Pi = linear_forest_precond_list_m(mtx, m, scale=scale)

    try:
        M = IterLinearOperator(mtx, [sparse.linalg.splu(P).solve for P in Pi], repeat_i=repeat_i)

    except RuntimeError:
        M = None

    return {
            's_coverage': None,
            's_degree'  : None,
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


# %%
def run_trial_precond(mtx, x, k_max_outer=10, k_max_inner=20, title=None, title_x=None, custom=None):
    """ Compare the performance of spanning tree preconditioners
    """
    mtx_is_symmetric = sparse_is_symmetric(mtx)
    sc = []
    sd = []
    labels = []
    preconds = []

    # Unpreconditioned system
    sc.append(1)
    sd.append(s_degree(mtx))
    preconds.append(None)
    labels.append('unpreconditioned')    

    # Jacobi preconditioner
    jacobi = trial_jacobi(mtx)
    sc.append(jacobi['s_coverage'])
    sd.append(jacobi['s_degree'])
    preconds.append(jacobi['precond'])
    labels.append('jacobi')

    # Tridiagonal preconditioner
    tridiag = trial_tridiag(mtx)
    sc.append(tridiag['s_coverage'])
    sd.append(tridiag['s_degree'])
    preconds.append(tridiag['precond'])
    labels.append('tridiag')

    # Maximum spanning tree preconditioner
    max_st = trial_max_st(mtx, mtx_is_symmetric)
    sc.append(max_st['s_coverage'])
    sd.append(max_st['s_degree'])
    preconds.append(max_st['precond'])
    labels.append('maxST')
    
    # # Maximum spanning tree preconditioner, applied to pruned matrix
    # max_st_pruned = trial_max_st(mtx, mtx_is_symmetric, prune=20)
    # sc.append(max_st_pruned['s_coverage'])
    # sd.append(max_st_pruned['s_degree'])
    # preconds.append(max_st_pruned['precond'])
    # labels.append('maxST (pruned A)')

    
    # # Maximum spanning tree preconditioner, additive factors (m = 2..5)
    # for m in range(2, 6):
    #     max_st_add_m = trial_max_st_add_m(mtx, mtx_is_symmetric, m)
    #     sc.append(max_st_add_m['s_coverage'])
    #     sd.append(max_st_add_m['s_degree'])
    #     preconds.append(max_st_add_m['precond'])
    #     labels.append(f'maxST+ (m = {m})')

    # # Maximum spanning tree preconditioner, multiplicative factors (m = 2..5)
    # for m in range(2, 6):
    #     max_st_mult_m = trial_max_st_mult_m(mtx, mtx_is_symmetric, m, scale=0.01)
    #     sc.append(max_st_mult_m['s_coverage'])
    #     sd.append(max_st_mult_m['s_degree'])
    #     preconds.append(max_st_mult_m['precond'])
    #     labels.append(f'maxST* (m = {m})')

    # # Maximum spanning tree preconditioner, inner alternating factors (m = 2..5)
    # for m in range(2, 6):
    #     max_st_alt_m_i = trial_max_st_alt_i(mtx, mtx_is_symmetric, m, scale=0.01)
    #     sc.append(max_st_alt_m_i['s_coverage'])
    #     sd.append(max_st_alt_m_i['s_degree'])
    #     preconds.append(max_st_alt_m_i['precond'])
    #     labels.append(f'maxST_alt_i (m = {m})')

    # # Maximum spanning tree preconditioner, outer alternating factors (m = 2..5)
    # for m in range(2, 6):
    #     max_st_alt_m_o = trial_max_st_alt_o(mtx, mtx_is_symmetric, 1, scale=0, repeat_i=m)
    #     #max_st_alt_m_o = trial_max_st_alt_o(mtx, mtx_is_symmetric, m, scale=0.01, repeat_i=0)
    #     sc.append(max_st_alt_m_o['s_coverage'])
    #     sd.append(max_st_alt_m_o['s_degree'])
    #     preconds.append(max_st_alt_m_o['precond'])
    #     labels.append(f'maxST_alt_o (m = {m})')
    
    # # Maximum spanning tree preconditioner, iterative inverses
    # for m in range(2, 6):
    #     max_st_inv_m = trial_max_st_inv_m(mtx, mtx_is_symmetric, m, prune=None)
    #     sc.append(max_st_inv_m['s_coverage'])
    #     sd.append(max_st_inv_m['s_degree'])
    #     preconds.append(max_st_inv_m['precond'])
    #     labels.append(f'max_ST_inv (m = {m})')

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

    # # Maximum linear forest preconditioner, multiplicative factors (m = 2..5),
    # # applied right-to-left
    # for m in range(2, 6):
    #     max_lf_mult_m = trial_max_lf_mult_m(mtx, mtx_is_symmetric, m, scale=0.01)
    #     sc.append(max_lf_mult_m['s_coverage'])
    #     sd.append(max_lf_mult_m['s_degree'])
    #     preconds.append(max_lf_mult_m['precond'])
    #     labels.append(f'maxLF* (m = {m})')
    
    # # Maximum spanning tree preconditioner, inner alternating factors (m = 2..5)
    # for m in range(2, 6):
    #     max_lf_alt_m_i = trial_max_lf_alt_i(mtx, mtx_is_symmetric, m, scale=0.01)
    #     sc.append(max_lf_alt_m_i['s_coverage'])
    #     sd.append(max_lf_alt_m_i['s_degree'])
    #     preconds.append(max_lf_alt_m_i['precond'])
    #     labels.append(f'maxLF_alt_i (m = {m})')

    # # Maximum spanning tree preconditioner, outer alternating factors (m = 2..5)
    # for m in range(2, 6):
    #     max_lf_alt_m_o = trial_max_lf_alt_o(mtx, mtx_is_symmetric, m, scale=0.01)
    #     sc.append(max_lf_alt_m_o['s_coverage'])
    #     sd.append(max_lf_alt_m_o['s_degree'])
    #     preconds.append(max_lf_alt_m_o['precond'])
    #     labels.append(f'maxLF_alt_o (m = {m})')
    
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
    fig1.savefig(f'{title}_x{title_x}.png', bbox_inches='tight')
    
    ax2.legend(title=f'{title}, x{title_x}')
    fig2.savefig(f'{title}_x{title_x}_fre.png', bbox_inches='tight')

    plt.close()
    

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

# %%
