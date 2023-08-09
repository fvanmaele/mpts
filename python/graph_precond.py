#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:39:46 2023

@author: archie
"""

import networkx as nx
import numpy as np
from scipy import sparse

from sparse_util  import sparse_prune, sparse_scale, sparse_max_n
from diag_precond import diagp1, unit


# %% Generalized graph preconditioner
# optG is assumed to not return diagonal elements in the adjacency matrix
def graph_normalized(mtx):
    """  Normalized and symmetrized version of a graph for segmentation
    """
    mtx_diagp_inv = sparse.diags(1 / diagp1(mtx))
    
    return nx.Graph(abs(mtx_diagp_inv @ mtx) + abs(mtx_diagp_inv @ mtx).T)
    

def graph_precond(mtx, optG):
    C = graph_normalized(mtx)
    D = sparse.diags(mtx.diagonal())  # DIAgonal
    O = optG(C)  # graph optimization function (spanning tree, linear forest, etc.)
    
    return sparse_prune(mtx, sparse.coo_array(nx.to_scipy_sparse_array(O) + D))


def graph_precond_add_m(mtx, optG, m):
    C = graph_normalized(mtx)

    # Begin with an empty sparse matrix
    S = sparse.csr_matrix(sparse.dok_matrix(mtx.shape))

    # Retrieve diagonal of input matrix for later adding
    D = sparse.diags(mtx.diagonal())

    # In every iteration, the optimized graph is computed for the remainder 
    # (excluding self-loops)
    for k in range(m-1):
        Mk = nx.to_scipy_sparse_array(optG(C))  # no diagonal elements
        
        # Accumulation of spanning trees (may have any amount of cycles)
        S = S + Mk
        
        # Subtract weights for the next iteration
        C = nx.Graph(nx.to_scipy_sparse_array(C) - Mk)

    return sparse_prune(mtx, sparse.coo_array(S + D))


def graph_precond_list_m(mtx, optG, m, scale=None):
    C = graph_normalized(mtx)
    M = nx.to_scipy_sparse_array(optG(C))  # no diagonal elements
    S = [M]

    # Retrieve diagonal of input matrix for later adding
    D = sparse.diags(mtx.diagonal())

    # Apply graph optimization to graph with entries scaled according to indices
    # retrieved in the previous step (2.2)
    for k in range(1, m):
        # C -> scale(C, S(M) \ S_diag(M), scale)
        C = nx.Graph(sparse_scale(nx.to_scipy_sparse_array(C).tocoo(), 
                                  sparse.coo_array(M), scale=scale))
        M = nx.to_scipy_sparse_array(optG(C))       
        S.append(M)

    return [sparse_prune(mtx, sparse.coo_array(Sk + D)) for Sk in S]
    

def graph_precond_mos_m(mtx, m, precond):
    n, _ = mtx.shape
    Id = sparse.eye(n)

    # Vector of weights, matching the number of non-zero elements in each row of A
    mtx_row_idx, _, _ = sparse.find(mtx)
    _, mtx_q = np.unique(mtx_row_idx, return_counts=True)

    # Check if B converges towards identity matrix
    B_diff = []  # TODO: add warning if B_diff gets "too large" in some iteration
    M_diff = []  # TODO: distance of B_l to the MOS preconditioner applied to A
    M_MOS  = []
    B = mtx.copy()

    for l in range(0, m):
        M = precond(B)  # includes diagonal of mtx1 (S^diag)
        B = sparse_max_n(B @ sparse.linalg.inv(M), mtx_q)

        B_diff.append(sparse.linalg.norm(Id - B))
        M_MOS.append(M.copy())

    return M_MOS, B_diff


def precond_mos_d(A, Al_pp, T, remainder=False):
    """
    Generalization of ILU factorizations (MOS Ansatz)

    Parameters
    ----------
    A : sparse.matrix
        Input coefficient matrix.
    Am : sparse.matrix...
        Off-diagonal matrices A''_0, ..., A''_{m-1}
    T : np.array
        Invertible diagonal matrix with unit(T) = unit(diag(A)) and |T| >= |diag(A)|
    remainder : bool
        If True, compute R = A - M_MOS_d (requires m-1 sparse matrix-matrix products)

    Returns
    -------
    M_MOS_d : sparse.matrix...
        MOS-d preconditioner as list of (multiplicative) factors

    """
    m = len(Al_pp)
    assert m >= 2

    # Precondition checks for diagonal matrix T
    A_diag = A.diagonal()
    assert T.ndim == 1              # diagonal matrix, 1d representation
    assert np.any(T > 0)            # invertible
    assert abs(T) >= abs(A_diag)    # |T| >= |diag(A)|
    assert np.all(unit(T) == unit(A_diag))
    
    # Intermediate diagonal factors (1.33)
    T_inv = 1 / T
    E = 1/m * T_inv * (A_diag - T)
    I = np.ones(len(T))
    assert np.all(E <= 0)
    assert np.allclose(T * (I + m*E), A_diag)
    
    # (1.34)
    J = I + E
    assert np.all((m - 1)/m * I <= J)
    assert np.all(J <= I)
    
    # Compute factors A' in dependence of A''_0,...,A''_{m-1}
    Al_p = []
    for l in range(0, m):
        Al_p.append(sparse.diags(J ** -(m-1-l)) @ Al_pp[l] @ sparse.diags(J ** -l))

    # Compute MOS-d preconditioner
    M_MOS_d = [sparse.diags(T)]
    for l in reversed(range(0, m)):
        M_MOS_d.append(sparse.diags(J) + sparse.diags(T_inv) @ Al_p[l])

    # Explicitly compute matrix-matrix products for remainder calculation
    R = None
    if remainder:
        M_prod = M_MOS_d[0]
        for k in reversed(range(1, m+1)):
            M_prod = M_prod @ M_MOS_d[k]
        # TODO: sparse condition estimate of A, |R|/|A| < 1/cond(A)
        R = M_prod - A
    
    return M_MOS_d, R


def graph_precond_mos_d(mtx, optG, m, scale=None, remainder=False):
    # Note: no permutation done here
    Al_pp = graph_precond_list_m(mtx, optG, m, scale=scale)
    
    return precond_mos_d(mtx, Al_pp, diagp1(mtx), remainder=remainder)
    

# %% Spanning tree preconditioner
def spanning_tree_precond(mtx):
    """ Compute spanning tree preconditioner for a given sparse matrix.
    """
    return graph_precond(mtx, nx.maximum_spanning_tree)


def spanning_tree_precond_add_m(mtx, m):
    """ Compute m spanning tree factors, computed by subtraction from the original graph.
    """
    return graph_precond_add_m(mtx, nx.maximum_spanning_tree, m)


def spanning_tree_precond_list_m(mtx, m, scale=None):
    """ Compute m spanning tree factors, computed by successively scaling of optimal entries
    """
    return graph_precond_list_m(mtx, nx.maximum_spanning_tree, m, scale=scale)


def spanning_tree_precond_mos_m(mtx, m):
    """ Compute spanning tree preconditioner iteratively, by computing (pruned) inverses
    """
    return graph_precond_mos_m(mtx, m, spanning_tree_precond)


def spanning_tree_precond_mos_d(mtx, m, scale=None, remainder=False):
    return graph_precond_mos_d(mtx, nx.maximum_spanning_tree, m, scale=scale, remainder=remainder)


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
    

def linear_forest_precond(mtx):
    """ Compute linear forest preconditioner for a given sparse matrix.
    """
    return graph_precond(mtx, linear_forest)


def linear_forest_precond_add_m(mtx, m):
    """ Compute m linear forest factors, computed by subtraction from the original graph.
    """
    return graph_precond_add_m(mtx, linear_forest, m)


def linear_forest_precond_list_m(mtx, m, scale=0):
    """ Compute m linear forest factors, computed by successively scaling of optimal entries
    """
    return graph_precond_list_m(mtx, linear_forest, m, scale=scale)


def linear_forest_precond_mos_m(mtx, m):
    """ Compute linear forest preconditioner iteratively, by computing (pruned) inverses
    """
    return graph_precond_mos_m(mtx, m, linear_forest_precond)


def linear_forest_precond_mos_d(mtx, m, scale=None, remainder=False):
    return graph_precond_mos_d(mtx, linear_forest, m, scale=scale, remainder=remainder)
