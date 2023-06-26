#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:22:49 2023

@author: archie
"""

import numpy as np
from scipy.sparse import issparse, find
from scipy.io import mmread
from pathlib import Path

def is_diagonally_dominant(matrix):
    if not issparse(matrix):
        raise ValueError("Input matrix must be a Scipy sparse matrix.")
    
    # Get the nonzero elements and their corresponding row indices
    data, rows, cols = find(matrix)
    
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Iterate over each row
    for i in range(matrix.shape[0]):
        row_values = data[rows == i]
        row_sum = np.sum(np.abs(row_values)) - np.abs(matrix.diagonal()[i])
        
        if np.abs(matrix.diagonal()[i]) < row_sum:
            return False
    
    return True


def main(mtx_path):
    mtx = mmread(mtx_path)
    dd = is_diagonally_dominant(mtx)
    if dd:
        print(f'matrix {mtx_path.stem} is diagonally dominant')
    else:
        print(f'matrix {mtx_path.stem} is not diagonally dominant')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='matrix', description='analyze properties of a matrix')
    parser.add_argument('mtx', type=str)
    
    args = parser.parse_args()
    main(Path(args.mtx))