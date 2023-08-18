#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 04:20:12 2023

@author: user
"""

import trial_precond as tr
import precond as pc
import networkx as nx

#from scipy.io import mmread, mmwrite
from scipy       import sparse
from precond     import diagp0, diagp1, diagp2
from sys         import stderr
from sparse_lops import AltLinearOperator, IterLinearOperator, ProdLinearOperator


# %%
def precond_setup(mtx):
    preconds = {}

    #print('setup: orig', file=stderr)
    preconds['orig'] = [tr.precond_orig(mtx)]
    
    #print('setup: jacobi', file=stderr)
    preconds['diagp0'] = [tr.precond_diag(mtx, diagp0(mtx))]
    preconds['diagp1'] = [tr.precond_diag(mtx, diagp1(mtx))]
    preconds['diagp2'] = [tr.precond_diag(mtx, diagp2(mtx))]
    
    #print('setup: tridiag', file=stderr)
    preconds['tridiag'] = [tr.precond_tridiag(mtx)]
    
    #print('setup: ilu0', file=stderr)
    preconds['ilu0'] = [tr.precond_ilu0(mtx)]
    
    return preconds


# %%
# TODO: save matrix factors as .mtx files, load if available
# TODO: print compute time for each setup phase
def precond_setup_graph(mtx, opt_graph, opt_label, m_max):
    """ Compare the performance of spanning tree preconditioners
    """
    preconds = {}
    m_range = range(2, m_max+1)

    # MST factors, scale = 0 (MOS-d) and scale = 0.01 (ALT-i, ALT-o)
    #print(f'setup: factors (mst, m = {m_max})', file=stderr)
    Pi_graph_noscale = pc.graph_precond_list_m(mtx, opt_graph, m_max, scale=0)
    Pi_graph_scale   = pc.graph_precond_list_m(mtx, opt_graph, m_max, scale=0.01)

    
    # Optimal graph preconditioner
    #print(f'setup: {opt_label}', file=stderr)
    P_graph = Pi_graph_noscale[0]
    preconds[opt_label] = [tr.precond_mtx(mtx, P_graph)]


    # Additive factors
    #print(f'setup: {opt_label}_add', file=stderr)
    preconds[opt_label + '_add'] = []
    
    for m in m_range:
        preconds[opt_label + '_add'].append(
            tr.precond_mtx(mtx, pc.graph_precond_add_m(mtx, opt_graph, m)))


    # MOS-a factors M_0, ..., M_{m-1}
    #print(f'setup: {opt_label}_mos-a', file=stderr)
    graph_mos_a, graph_mos_a_diff = pc.graph_precond_mos_a(mtx, opt_graph, m_max)
    preconds[opt_label + '_mos-a'] = []
    
    for m in m_range:
        preconds[opt_label + '_mos-a'].append(
            tr.precond_lops(mtx, [sparse.diags(diagp1(mtx)).tocoo()] + graph_mos_a[:m], ProdLinearOperator))


    # MOS-d factors M''_0, .., M''_{m-1}
    #print(f'setup: {opt_label}_mos-d', file=stderr)
    graph_mos_d = pc.graph_precond_mos_d(mtx, Pi_graph_noscale, diagp1(mtx))
    preconds[opt_label + '_mos-d'] = []
    
    for m in m_range:
        preconds[opt_label + '_mos-d'].append(
            tr.precond_lops(mtx, [sparse.diags(diagp1(mtx)).tocoo()] + graph_mos_d[:m], ProdLinearOperator))


    # Inner alternating factors
    #print(f'setup: {opt_label}_alt-i', file=stderr)
    preconds[opt_label + '_alt-i'] = []

    for m in m_range:
        preconds[opt_label + '_alt-i'].append(
            tr.precond_lops(mtx, Pi_graph_scale[:m], AltLinearOperator))


    # Outer alternating factors
    #print(f'setup: {opt_label}_alt-o', file=stderr)
    preconds[opt_label + '_alt-o'] = []
    
    for m in m_range:
        preconds[opt_label + '_alt-o'].append(
            tr.precond_lops(mtx, Pi_graph_scale[:m], IterLinearOperator))


    # Outer repeating factors
    #print(f'setup: {opt_label}_alt-o-repeat', file=stderr)
    preconds[opt_label + '_alt-o-repeat'] = []
    
    for m in m_range:
        preconds[opt_label + '_alt-o-repeat'].append(
            tr.precond_lops(mtx, [P_graph], IterLinearOperator, repeat_i=m))

    return preconds


def precond_setup_mst(mtx, m_max):
    return precond_setup_graph(mtx, nx.maximum_spanning_tree, 'max-st', m_max)


def precond_setup_lf(mtx, m_max):
    return precond_setup_graph(mtx, pc.linear_forest, 'max-lf', m_max)
