#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fvanmaele
"""

# %%
from pathlib import Path
from copy import copy

import numpy as np
import networkx as nx

from scipy.io import mmread
from scipy import sparse


# %%
def sparse_is_symmetric(mtx, tol=1e-10):
    """Check a matrix for numerical symmetry
    """
    assert sparse.issparse(mtx)

    return (abs(mtx-mtx.T) > tol).nnz == 0


def find_n_factors(mtx, n):
    """Sequential greedy [0,n]-factor computation on a weighted graph G = (V, E)
    """
    assert sparse_is_symmetric(mtx)
    assert n > 0

    # factors = [[] for _ in range(0, mtx.shape[0])]
    G_abs = nx.Graph(abs(mtx))  # absolute values for edge weights
    G_abs.remove_edges_from(nx.selfloop_edges(G_abs))  # ignore self-loops (diagonal elements)

    Gn = nx.Graph()
    Gn.add_nodes_from(G_abs)
    
    # Iterate over edges in decreasing weight order
    for v, w, O in sorted(G_abs.edges(data=True), key=lambda t: t[2].get('weight', 1), reverse=True):
        assert(v != w)  # sanity check

        # Empty set, one vertex, ..., or n vertices from the neighborhood of a vertex v
        if Gn.degree[v] < n and Gn.degree[w] < n:
            Gn.add_edge(v, w, weight=O['weight'])

        # if len(factors[v]) < n and len(factors[w]) < n and v != w:
        #     factors[v].append(w)  # undirected graph
        #     factors[w].append(v)

    return Gn  # does not include loops


def linear_forest(mtx):
    Gn = find_n_factors(mtx, 2)
    forest = nx.Graph()
    forest.add_nodes_from(Gn)

    for c in nx.connected_components(Gn):
        Gc = Gn.subgraph(c).copy()
        try:
            nx.find_cycle(Gn.subgraph(c))
            # TODO: Break cycle by removing edge of smallest weight
            
        except nx.NetworkXNoCycle:
            # OK, add to forest
            forest.update(Gc)

    return forest


# TODO: permute (adjacency matrix of) linear forest to tridiagonal form
def linear_forest_permute(mtx):
    pass