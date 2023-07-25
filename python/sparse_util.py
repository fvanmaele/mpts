#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:36:49 2023

@author: archie
"""
import warnings
import numpy as np
from scipy import sparse


def sparse_is_symmetric(mtx, tol=1e-10):
    """ Check a matrix for numerical symmetry
    """
    # XXX: does networkx use any tolerances when constructing an (undirected) graph from a matrix?
    return (abs(mtx-mtx.T) > tol).nnz == 0


# TODO: return percentage of diagonally dominant rows
def sparse_is_ddiag(matrix):
    if not sparse.issparse(matrix):
        raise ValueError("Input matrix must be a Scipy sparse matrix.")
    
    # Get the nonzero elements and their corresponding row indices
    data, rows, cols = sparse.find(matrix)
    
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Iterate over each row
    rows_ddiag = 0
    for i in range(matrix.shape[0]):
        row_values = data[rows == i]
        row_sum = np.sum(np.abs(row_values)) - np.abs(matrix.diagonal()[i])
        
        if np.abs(matrix.diagonal()[i]) >= row_sum:
            rows_ddiag += 1
    
    return rows_ddiag / matrix.shape[0]


def sparse_mask(mtx, mtx_mask, scale=None):
    """ Index a sparse matrix with another sparse matrix of the same dimension.
    
    It is assumed that non-zero indices of `mtx_mask` are a subset of 
    non-zero indices of `mtx` (with potentially differing entries).

    If the `scale` parameter is specified, indexed entries are scaled with a given
    factor instead of removed.
    """
    # TODO: do a precondition check on "is a subset of"
    assert mtx.shape == mtx_mask.shape
    assert sparse.issparse(mtx)
    assert sparse.issparse(mtx_mask)

    rows = []
    cols = []
    data = []
    mask = {}

    for i, j, v in zip(mtx_mask.row, mtx_mask.col, mtx_mask.data):
        mask[(i, j)] = True

    for i, j, v in zip(mtx.row, mtx.col, mtx.data):
        if scale is not None:
            rows.append(i)
            cols.append(j)
            
            if (i, j) in mask:
                data.append(scale*v)
            else:
                data.append(v)
        else:
            if (i, j) in mask:
                rows.append(i)
                cols.append(j)

                data.append(v)

    return sparse.coo_matrix((data, (rows, cols)))


def s_coverage(mtx, mtx_pruned):
    """ Compute the S-coverage as quality measure for a sparse preconditioner
    """
    assert sparse.issparse(mtx)
    assert sparse.issparse(mtx_pruned)
    
    sc = abs(sparse.csr_matrix(mtx_pruned)).sum() / abs(sparse.csr_matrix(mtx)).sum()
    if sc > 1:
        warnings.warn('S coverage is greater than 1')

    return sc


def s_degree(mtx):
    """Compute the maximum degree for the graph of an adjacency matrix.
    
    Self-loops (diagonal elements) are ignored.
    """
    assert sparse.issparse(mtx)
    
    M = sparse.csr_matrix(mtx - sparse.diags(mtx.diagonal()))
    M_deg = [M.getrow(i).getnnz() for i in range(M.shape[0])]

    return max(M_deg)


def prune_sparse_matrix(matrix, N):
    #assert sparse.isspmatrix_csr(matrix)
    # Find the nonzero elements and their corresponding row indices
    row_indices, col_indices, values = sparse.find(matrix)

    # Iterate over the unique row indices
    unique_row_indices = np.unique(row_indices)

    for row_idx in unique_row_indices:
        # Find the indices of nonzero elements in the current row
        row_indices_mask = (row_indices == row_idx)
        row_values = values[row_indices_mask]
        row_col_indices = col_indices[row_indices_mask]

        if len(row_values) <= N:
            continue

        # Sort the row values by their absolute values
        sorted_indices = np.argsort(np.abs(row_values))

        # Prune the row values and indices beyond N
        pruned_indices = row_col_indices[sorted_indices[:-N]]
        #pruned_values = row_values[sorted_indices[:-N]]

        # Set the pruned values and indices in the matrix
        matrix[row_idx, pruned_indices] = 0
        #matrix[row_idx, pruned_indices] = pruned_values

    return matrix.tocoo()
