#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 04:20:12 2023

@author: user
"""

#from scipy.io import mmread, mmwrite
from scipy import sparse
from diag_precond import diagp0, diagp1, diagp2

import trial_precond as tr
import graph_precond as gp
import networkx as nx

from sparse_lops import AltLinearOperator, IterLinearOperator


# %%
def setup_precond(mtx):
    preconds = {}

    #print('setup: orig')
    preconds['orig'] = [tr.precond_orig(mtx)]
    
    #print('setup: jacobi')
    preconds['diagp0'] = [tr.precond_diag(mtx, diagp0(mtx))]
    preconds['diagp1'] = [tr.precond_diag(mtx, diagp1(mtx))]
    preconds['diagp2'] = [tr.precond_diag(mtx, diagp2(mtx))]
    
    #print('setup: tridiag')
    preconds['tridiag'] = [tr.precond_tridiag(mtx)]
    
    #print('setup: ilu0')
    preconds['ilu0'] = [tr.precond_ilu0(mtx)]
    
    return preconds


# %%
# TODO: save matrix factors as .mtx files, load if available
# TODO: print compute time for each setup phase
def setup_precond_graph(mtx, opt_graph, opt_label, m_max):
    """ Compare the performance of spanning tree preconditioners
    """
    preconds = {}
    m_range = range(2, m_max+1)

    # MST factors, scale = 0 (MOS-d) and scale = 0.01 (ALT-i, ALT-o)
    #print(f'setup: factors (mst, m = {m_max})')
    Pi_graph_noscale = gp.graph_precond_list_m(mtx, opt_graph, m_max, scale=0)
    Pi_graph_scale   = gp.graph_precond_list_m(mtx, opt_graph, m_max, scale=0.01)

    # Optimal graph preconditioner
    #print(f'setup: {opt_label}')
    P_graph = Pi_graph_noscale[0]
    preconds[opt_label] = [tr.precond_mtx(mtx, P_graph)]


    # Additive factors
    #print(f'setup: {opt_label}_add')
    preconds[opt_label + '_add'] = []
    
    for m in m_range:
        preconds[opt_label + '_add'].append(
            tr.precond_mtx(mtx, gp.graph_precond_add_m(mtx, opt_graph, m))
        )


    # MOS-a factors
    #print(f'setup: {opt_label}_mos-a')
    graph_mos_a, graph_mos_a_diff = gp.graph_precond_mos_a(mtx, opt_graph, m_max)
    preconds[opt_label + '_mos-a'] = []
    
    for m in m_range:
        preconds[opt_label + '_mos-a'].append(
            tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)).tocoo()] + graph_mos_a[:m])
        )


    # MOS-d factors
    #print(f'setup: {opt_label}_mos-d')
    graph_mos_d = gp.graph_precond_mos_d(mtx, Pi_graph_noscale, diagp1(mtx))
    preconds[opt_label + '_mos-d'] = []
    
    for m in m_range:
        preconds[opt_label + '_mos-d'].append(
            tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)).tocoo()] + graph_mos_d[:m])
        )


    # Inner alternating factors
    #print(f'setup: {opt_label}_alt-i')
    preconds[opt_label + '_alt-i'] = []

    for m in m_range:
        preconds[opt_label + '_alt-i'].append(
            tr.precond_lops(mtx, Pi_graph_scale[:m], AltLinearOperator)
        )


    # Outer alternating factors
    #print(f'setup: {opt_label}_alt-o')
    preconds[opt_label + '_alt-o'] = []
    
    for m in m_range:
        preconds[opt_label + '_alt-o'].append(
            tr.precond_lops(mtx, Pi_graph_scale[:m], IterLinearOperator)
        )


    # Outer repeating factors
    #print(f'setup: {opt_label}_alt-o-repeat')
    preconds[opt_label + '_alt-o-repeat'] = []
    
    for m in m_range:
        preconds[opt_label + '_alt-o-repeat'].append(
            tr.precond_lops(mtx, [P_graph], IterLinearOperator, repeat_i=m)
        )

    return preconds


# %%
def setup_precond_mst(mtx, m_max):
    return setup_precond_graph(mtx, nx.maximum_spanning_tree, 'max-st', m_max)

def setup_precond_lf(mtx, m_max):
    return setup_precond_graph(mtx, gp.linear_forest, 'max-lf', m_max)
