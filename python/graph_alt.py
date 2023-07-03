#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:10:16 2023

@author: archie
"""

def trial_max_st_alt_m(mtx, mtx_is_symmetric, m):
    assert m > 1
    
    if not mtx_is_symmetric:
        Pi = spanning_tree_precond_add_m(mtx, m, tolist=True, symmetrize=True)
    else:
        Pi = spanning_tree_precond_add_m(mtx, m, tolist=True)

    try:
        M = AltLinearOperator(mtx.shape, [sparse.linalg.splu(P).solve for P in Pi])
    except RuntimeError:
        M = None

    return {
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


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


def trial_max_st_mult_m(mtx, mtx_is_symmetric, m, reverse=False):
    assert m > 1
    
    if not mtx_is_symmetric:
        Pi = spanning_tree_precond_add_m(mtx, m, tolist=True, symmetrize=True)
    else:
        Pi = spanning_tree_precond_add_m(mtx, m, tolist=True)

    try:
        # Create sparse operator which applies Mi successively
        if reverse:
            f = lambda x: solve_ltr(x, [sparse.linalg.splu(P).solve for P in Pi])
        else:
            f = lambda x: solve_rtl(x, [sparse.linalg.splu(P).solve for P in Pi])

        M = sparse.linalg.LinearOperator(mtx.shape, f)

    except RuntimeError:
        M = None

    return { 
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


def trial_max_lf_alt_m(mtx, mtx_is_symmetric, m):
    assert m > 1
    
    if not mtx_is_symmetric:
        Pi = linear_forest_precond_add_m(mtx, m, tolist=True, symmetrize=True)
    else:
        Pi = linear_forest_precond_add_m(mtx, m, tolist=True)

    try:
        M = AltLinearOperator(mtx.shape, [sparse.linalg.splu(P).solve for P in Pi])
    except RuntimeError:
        M = None

    return {
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


def trial_max_lf_mult_m(mtx, mtx_is_symmetric, m, reverse=False):
    assert m > 1
    
    if not mtx_is_symmetric:
        Pi = linear_forest_precond_add_m(mtx, m, tolist=True, symmetrize=True)
    else:
        Pi = linear_forest_precond_add_m(mtx, m, tolist=True)

    try:
        # Create sparse operator which applies Mi successively
        if reverse:
            f = lambda x: solve_ltr(x, [sparse.linalg.splu(P).solve for P in Pi])
        else:
            f = lambda x: solve_rtl(x, [sparse.linalg.splu(P).solve for P in Pi])

        M = sparse.linalg.LinearOperator(mtx.shape, f)

    except RuntimeError:
        M = None

    return { 
        's_coverage': None,
        's_degree'  : None,
        'precond'   : M
    }


# %%
# Maximum spanning tree preconditioner, multiplicative factors (m = 2..5),
# applied right-to-left
for m in range(2, 6):
    max_st_mult_m = trial_max_st_mult_m(mtx, mtx_is_symmetric, m, reverse=True)
    sc.append(max_st_mult_m['s_coverage'])
    sd.append(max_st_mult_m['s_degree'])
    preconds.append(max_st_mult_m['precond'])
    labels.append(f'maxST* (m = {m})')

# # Maximum spanning tree preconditioner, alternating factors (m = 2..5)
# for m in range(2, 6):
#     max_st_alt_m = trial_max_st_alt_m(mtx, mtx_is_symmetric, m)
    
#     sc.append(max_st_alt_m['s_coverage'])
#     sd.append(max_st_alt_m['s_degree'])
#     preconds.append(max_st_alt_m['precond'])
#     labels.append(f'maxST_alt (m = {m})')

# # Maximum linear forest preconditioner, multiplicative factors (m = 2..5),
# # applied right-to-left
# for m in range(2, 6):
#     max_lf_mult_m = trial_max_lf_mult_m(mtx, mtx_is_symmetric, m)
#     sc.append(max_lf_mult_m['s_coverage'])
#     sd.append(max_lf_mult_m['s_degree'])
#     preconds.append(max_lf_mult_m['precond'])
#     labels.append(f'maxLF* (m = {m})')

# # Maximum spanning tree preconditioner, alternating factors (m = 2..5)
# for m in range(2, 6):
#     max_lf_alt_m = trial_max_lf_alt_m(mtx, mtx_is_symmetric, m)
#     sc.append(max_lf_alt_m['s_coverage'])
#     sd.append(max_lf_alt_m['s_degree'])
#     preconds.append(max_lf_alt_m['precond'])
#     labels.append(f'maxLF_alt (m = {m})')
