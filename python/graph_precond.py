#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:39:46 2023

@author: archie
"""

import networkx as nx
from scipy import sparse
from math import ceil

from sparse_util import sparse_mask, prune_sparse_matrix


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


def graph_precond_list_m(mtx, optG, m, symmetrize=False, scale=None):
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


# TODO: specify a sparsity pattern (here: pruned powers / neumann expansion of coefficient matrix A)
def spanning_tree_precond_mos_m(mtx, m, symmetrize=False):
    """ Compute spanning tree preconditioner iteratively, by computing (pruned) inverses
    """
    n, _ = mtx.shape
    Id = sparse.eye(n)
    mtx_avg_deg = ceil(mtx.getnnz() / n)
    B = mtx.copy()

    B_diff = []  # TODO: add warning if B_diff gets "too large" in some iteration
    M_diff = []  # TODO: distance of B_l to the MOS preconditioner applied to A
    M_MOS  = []

    for l in range(0, m):
        M = spanning_tree_precond(B)  # includes diagonal of mtx1 (S^diag)
        B = sparse_mask((B @ sparse.linalg.inv(M)).tocoo(), prune_sparse_matrix(B @ B, mtx_avg_deg))  # neumann expansion
        
        B_diff.append(sparse.linalg.norm(Id - B))
        M_MOS.append(M.copy())

    return M_MOS


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


# def linear_forest_precond_mos_m(mtx, m, symmetrize=False):
#     """ Compute linear forest preconditioner iteratively, by computing (pruned) inverses
#     """
#     return graph_precond_inv_m(mtx, m, nx.maximum_spanning_tree, symmetrize=symmetrize)
