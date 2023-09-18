#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 19:20:15 2023

@author: archie
"""

from scipy         import sparse
from pathlib       import Path
from scipy.io      import mmread
from sparse_util   import s_coverage, s_degree, sparse_ddiag
from precond       import spanning_tree_precond, linear_forest_precond
from precond_diag  import diagp0


# %%
def main(mtx_l, normalize_diag):
    print('name,diag,max_lf,max_st,diag_dom,max_deg')

    for mtx_str in mtx_l:
        mtx  = sparse.coo_matrix(mmread(mtx_str))
        n, m = mtx.shape
        if n != m:
            continue

        mtx_name   = Path(mtx_str).stem
        mtx_mdeg   = s_degree(mtx)
        mtx_dd     = sparse_ddiag(mtx)
        mtx_diag   = sparse.diags(diagp0(mtx))
        
        try:
            mtx_max_lf = linear_forest_precond(mtx)
            mtx_max_st = spanning_tree_precond(mtx)
        except ValueError:
            continue

        cvg_diag   = s_coverage(mtx, mtx_diag,   normalize_diag=normalize_diag)
        cvg_max_lf = s_coverage(mtx, mtx_max_lf, normalize_diag=normalize_diag)
        cvg_max_st = s_coverage(mtx, mtx_max_st, normalize_diag=normalize_diag)

        print(f'{mtx_name},{cvg_diag},{cvg_max_lf},{cvg_max_st},{mtx_dd},{mtx_mdeg}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='matrix', description='analyze properties of a matrix')
    parser.add_argument('--normalize-diag', action='store_true',
                        help='multiply with diagp1 from the left (s-coverage)')
    parser.add_argument('mtx', nargs='+')
    
    args = parser.parse_args()
    main(args.mtx, args.normalize_diag)