#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 04:20:12 2023

@author: user
"""

from scipy.io import mmread, mmwrite
from scipy import sparse
from diag_precond import diagp1

import trial_precond as tr
import graph_precond as gp

from sparse_lops import AltLinearOperator, IterLinearOperator


# %%
def setup_precond(mtx):
    preconds = {}

    print('setup: orig')
    preconds['orig'] = tr.precond_orig(mtx)
    
    print('setup: jacobi')
    preconds['jacobi'] = tr.precond_jacobi(mtx)
    
    print('setup: tridiag')
    preconds['tridiag'] = tr.precond_tridiag(mtx)
    
    print('setup: ilu0')
    preconds['ilu0'] = tr.precond_ilu0(mtx)
    
    return preconds


# %%
# TODO: save matrix factors as .mtx files, load if available
# TODO: print compute time for each setup phase
def setup_precond_mst(mtx, m_max):
    """ Compare the performance of spanning tree preconditioners
    """
    preconds = {}

    # MST factors, scale = 0 (MOS-d) and scale = 0.01 (ALT-i, ALT-o)
    print('setup: factors (mst, m = {m})')
    Pi_max_st_noscale = gp.spanning_tree_precond_list_m(mtx, m_max, scale=0)
    Pi_max_st_scale   = gp.spanning_tree_precond_list_m(mtx, m_max, scale=0.01)

    # Maximum spanning tree preconditioner
    print('setup: mst')
    P_max_st = Pi_max_st_noscale[0]
    preconds['max_st'] = tr.precond_graph(mtx, P_max_st)


    # Additive factors
    print('setup: mst_add')
    preconds['max_st_add'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_add'].append(tr.precond_graph(mtx, gp.spanning_tree_precond_add_m(mtx, m)))


    # MOS-a factors
    print('setup: mst_mos_a')
    max_st_mos_a, max_st_mos_a_diff = gp.spanning_tree_precond_mos_a(mtx, m_max)
    preconds['max_st_mos_a'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_mos_a'].append(tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)).tocoo()] + max_st_mos_a[:m]))


    # MOS-d factors
    print('setup: mst_mos_d')
    max_st_mos_d = gp.graph_precond_mos_d(mtx, Pi_max_st_noscale, diagp1(mtx))
    preconds['max_st_mos_d'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_mos_d'].append(tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)).tocoo()] + max_st_mos_d[:m]))


    # Inner alternating factors
    print('setup: mst_alt_i')
    preconds['max_st_alt_i'] = []

    for m in range(2, m_max+1):
        preconds['max_st_alt_i'].append(tr.precond_lops(mtx, Pi_max_st_scale[:m], AltLinearOperator))


    # Outer alternating factors
    print('setup: mst_alt_o')
    preconds['max_st_alt_o'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_alt_o'].append(tr.precond_lops(mtx, Pi_max_st_scale[:m], IterLinearOperator))


    # Outer repeating factors
    print('setup: mst_alt_o_repeat')
    preconds['max_st_alt_o_repeat'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_alt_o_repeat'].append(tr.precond_lops(mtx, [P_max_st], IterLinearOperator, repeat_i=m))

    return preconds


# %%
# TODO: save matrix factors as .mtx files, load if available
# XXX: same logic as setup_precond_mst(), take optimization function/label as parameter
def setup_precond_lf(mtx, m_max):
    preconds = {}
    
    # LF factors, scale = 0 (MOS-d) and scale = 0.01 (ALT-i, ALT-o)
    print('setup: factors (lf, m = {m})')
    Pi_max_lf_noscale = gp.linear_forest_precond_list_m(mtx, m_max, scale=0)
    Pi_max_lf_scale   = gp.linear_forest_precond_list_m(mtx, m_max, scale=0.01)
    
    # Maximum linear forest preconditioner
    print('setup: lf')
    P_max_lf = Pi_max_lf_noscale[0]
    preconds['max_lf'] = tr.precond_graph(mtx, P_max_lf)


    # Additive factors
    print('setup: lf_add')
    preconds['max_lf_add'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_add'].append(tr.precond_graph(mtx, gp.linear_forest_precond_add_m(mtx, m)))
    
    
    # MOS-a factors
    print('setup: lf_mos_a')
    max_lf_mos_a, max_lf_mos_a_diff = gp.linear_forest_precond_mos_a(mtx, m_max)
    preconds['max_lf_mos_a'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_mos_a'].append(tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)).tocoo()] + max_lf_mos_a[:m]))
    
    
    # MOS-d factors
    print('setup: lf_mos_d')
    max_lf_mos_d = gp.graph_precond_mos_d(mtx, Pi_max_lf_noscale, diagp1(mtx))
    preconds['max_lf_mos_d'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_mos_d'].append(tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)).tocoo()] + max_lf_mos_d[:m]))
    
    
    # Inner alternating factors
    print('setup: lf_alt_i')
    preconds['max_lf_alt_i'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_alt_i'].append(tr.precond_lops(mtx, Pi_max_lf_scale[:m], AltLinearOperator))
    
    
    # Outer alternating factors
    print('setup: lf_alt_o')
    preconds['max_lf_alt_o'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_alt_o'].append(tr.precond_lops(mtx, Pi_max_lf_scale[:m], IterLinearOperator))
    
    
    # Outer repeating factors
    print('setup: lf_alt_o_repeat')
    preconds['max_lf_alt_o_repeat'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_alt_o_repeat'].append(tr.precond_lops(mtx, [P_max_lf], IterLinearOperator, repeat_i=m))

    return preconds