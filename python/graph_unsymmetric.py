#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 21:07:42 2023


@author: archie
"""

# %%
import networkx as nx


# %%
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
