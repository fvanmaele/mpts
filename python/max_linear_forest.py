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
    assert sparse.issparse(mtx)

    return (abs(mtx-mtx.T) > tol).nnz == 0


"""Sequential greedy [0,n]-factor computation on a weighted graph G = (V, E)
"""
def find_n_factors(mtx, n):
    assert sparse_is_symmetric(mtx)
    assert n > 0

    # factors = [[] for _ in range(0, mtx.shape[0])]
    G  = nx.Graph(abs(mtx))                    # absolute values for edge weights
    G.remove_edges_from(nx.selfloop_edges(G))  # ignore self-loops (diagonal elements)

    Gn = nx.Graph()
    Gn.add_nodes_from(G)
    
    # Iterate over edges in decreasing weight order
    for v, w, O in sorted(G.edges(data=True), key=lambda t: t[2].get('weight', 1), reverse=True):
        assert(v != w)

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

    # # Break cycles by removing edge of smallest (absolute) weight
    # for v, nbh in enumerate(factors):
    #     if len(nbh) < 2:
    #         forest[v] = copy(nbh)

    #     elif nbh[0] in factors[nbh[1]]:
    #         # cycle found, remove smallest of three edges
    #         w = [G_abs.get_edge_data(v, nbh[0])['weight'],
    #              G_abs.get_edge_data(v, nbh[1])['weight'],
    #              G_abs.get_edge_data(nbh[0], nbh[1])['weight']
    #         ]
    #         amin = np.argmin(w)
    #         if amin == 0:
    #             forest[v] = [nbh[1]]
    #         elif amin == 1:
    #             forest[v] = [nbh[0]]
    #         elif amin == 2:
    #             pass

    #     else:
    #         # no cycle found, add path
    #         forest[v] = copy(nbh)

    # # XXX: return graph
    # return forest