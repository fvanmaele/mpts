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

    preconds['orig']    = tr.precond_orig(mtx)
    preconds['jacobi']  = tr.precond_jacobi(mtx)
    preconds['tridiag'] = tr.precond_tridiag(mtx)
    preconds['ilu0']    = tr.precond_ilu0(mtx)
    
    return preconds


# %%
# TODO: save matrix factors as .mtx files, load if available
# TODO: print compute time for each setup phase
def setup_precond_mst(mtx, m_max):
    """ Compare the performance of spanning tree preconditioners
    """
    preconds = {}

    # MST factors, scale = 0 (MOS-d) and scale = 0.01 (ALT-i, ALT-o)
    Pi_max_st_noscale = gp.spanning_tree_precond_list_m(mtx, m_max, scale=0)
    Pi_max_st_scale   = gp.spanning_tree_precond_list_m(mtx, m_max, scale=0.01)

    # Maximum spanning tree preconditioner
    P_max_st = Pi_max_st_noscale[0]
    preconds['max_st'] = tr.precond_graph(mtx, P_max_st)


    # Additive factors
    preconds['max_st_add'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_add'].append(tr.precond_graph(mtx, gp.spanning_tree_precond_add_m(mtx, m)))


    # MOS-a factors
    max_st_mos_a = gp.spanning_tree_precond_mos_a(mtx, m_max)
    preconds['max_st_mos_a'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_mos_a'].append(tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)), *max_st_mos_a[:m]]))


    # MOS-d factors
    max_st_mos_d = gp.graph_precond_mos_d(mtx, Pi_max_st_noscale, diagp1(mtx))
    preconds['max_st_mos_d'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_mos_d'].append(tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)), *max_st_mos_d[:m]]))


    # Inner alternating factors
    preconds['max_st_alt_i'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_alt_i'].append(tr.precond_lops(mtx, Pi_max_st_scale[:m], IterLinearOperator))


    # Outer alternating factors
    preconds['max_st_alt_o'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_alt_o'].append(tr.precond_lops(mtx, Pi_max_st_scale[:m], AltLinearOperator))


    # Outer repeating factors
    preconds['max_st_alt_o_repeat'] = []
    
    for m in range(2, m_max+1):
        preconds['max_st_alt_o_repeat'].append(tr.precond_lops(mtx, P_max_st, AltLinearOperator, repeat_i=m))

    return preconds


# %%
# TODO: save matrix factors as .mtx files, load if available
# XXX: same logic as setup_precond_mst(), take optimization function/label as parameter
def setup_precond_lf(mtx, m_max):
    preconds = {}
    
    # LF factors, scale = 0 (MOS-d) and scale = 0.01 (ALT-i, ALT-o)
    Pi_max_lf_noscale = gp.linear_forest_precond_list_m(mtx, m_max, scale=0)
    Pi_max_lf_scale   = gp.linear_forest_precond_list_m(mtx, m_max, scale=0.01)
    
    # Maximum linear forest preconditioner
    P_max_lf = Pi_max_lf_noscale[0]
    preconds['max_lf'] = tr.precond_graph(mtx, P_max_lf)


    # Additive factors
    preconds['max_lf_add'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_add'].append(tr.precond_graph(mtx, gp.linear_forest_precond_add_m(mtx, m)))
    
    
    # MOS-a factors
    max_lf_mos_a = gp.linear_forest_precond_mos_a(mtx, m_max)
    preconds['max_lf_mos_a'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_mos_a'].append(tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)), *max_lf_mos_a[:m]]))
    
    
    # MOS-d factors
    max_lf_mos_d = gp.graph_precond_mos_d(mtx, Pi_max_lf_noscale, diagp1(mtx))
    preconds['max_lf_mos_d'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_mos_d'].append(tr.precond_prod_r(mtx, [sparse.diags(diagp1(mtx)), *max_lf_mos_d[:m]]))
    
    
    # Inner alternating factors
    preconds['max_lf_alt_i'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_alt_i'].append(tr.precond_lops(mtx, Pi_max_lf_scale[:m], IterLinearOperator))
    
    
    # Outer alternating factors
    preconds['max_lf_alt_o'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_alt_o'].append(tr.precond_lops(mtx, Pi_max_lf_scale[:m], AltLinearOperator))
    
    
    # Outer repeating factors
    preconds['max_lf_alt_o_repeat'] = []
    
    for m in range(2, m_max+1):
        preconds['max_lf_alt_o_repeat'].append(tr.precond_lops(mtx, P_max_lf, AltLinearOperator, repeat_i=m))

    return preconds