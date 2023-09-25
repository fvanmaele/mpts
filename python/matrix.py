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
from precond_diag  import diagp0, diagp1

import numpy as np

# %%
def main(mtx_l):
    print('name,diag,max_lf,max_st,diag_dom,max_deg')

    for mtx_str in mtx_l:
        mtx  = sparse.coo_matrix(mmread(mtx_str))
        n, m = mtx.shape
        
        # Skip rectangular matrices
        if n != m:
            continue

        # Skip matrices with zero rows (<=> diagp1 is not invertible)
        mtx_diagp1 = diagp1(mtx)
        if not np.all(mtx_diagp1):
            continue

        # For purposes of comparing the S-coverage (between linear forest and
        # maximum spanning tree), we first normalize the matrix by multiplying
        # with the inverse of diagp_1.
        mtx = (sparse.diags(1 / mtx_diagp1) @ mtx).tocoo()

        # Compute matrix properties
        mtx_name   = Path(mtx_str).stem
        mtx_mdeg   = s_degree(mtx)              # maximum degree of corresponding graph (without loops)
        mtx_dd     = sparse_ddiag(mtx)          # diagonally dominant rows
        mtx_diag   = sparse.diags(diagp0(mtx))  # diagonal component

        # Compute graph preconditioner
        try:
            mtx_max_lf = linear_forest_precond(mtx)
            mtx_max_st = spanning_tree_precond(mtx)
        except ValueError:
            continue

        cvg_diag   = s_coverage(mtx, mtx_diag)
        cvg_max_lf = s_coverage(mtx, mtx_max_lf)
        cvg_max_st = s_coverage(mtx, mtx_max_st)

        print(f'{mtx_name},{cvg_diag},{cvg_max_lf},{cvg_max_st},{mtx_dd},{mtx_mdeg}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='matrix', description='analyze properties of a matrix')
    parser.add_argument('mtx', nargs='+')
    
    args = parser.parse_args()
    main(args.mtx)