#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:22:49 2023

@author: archie
"""

from scipy.io import mmread
from pathlib import Path

from sparse_util import sparse_is_ddiag

# %%
def main(mtx_path):
    mtx = mmread(mtx_path)
    dd  = sparse_is_ddiag(mtx)
    
    print(f'matrix {mtx_path.stem}: {dd*100}% diagonally dominant')


# %%
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='matrix', description='analyze properties of a matrix')
    parser.add_argument('mtx', type=str)
    
    args = parser.parse_args()
    main(Path(args.mtx))