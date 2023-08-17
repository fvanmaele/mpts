#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:52:00 2023

@author: archie
"""

import numpy as np


# %% Diagonal preconditioners
def unit(x):
    return np.where(x != 0, x / np.abs(x), 1)

def diagp0(mtx):
    return mtx.diagonal()

def row_sums_excluding_diagonal(mtx):
    # Set the diagonal elements to zero in the CSR matrix
    mtx = mtx.copy()
    mtx.setdiag(0)

    # Compute the sum of row elements excluding the diagonal elements
    return np.array(mtx.sum(axis=1)).flatten()

def diagp1(mtx):
    mtx_d = mtx.diagonal()
    return np.multiply(unit(mtx_d), np.maximum(np.abs(mtx_d), row_sums_excluding_diagonal(abs(mtx))))

def diagp2(mtx):
    mtx_d = mtx.diagonal()
    return np.multiply(unit(mtx_d), np.array(abs(mtx).sum(axis=1)).flatten())

def diagl1(mtx):
    mtx_d = mtx.diagonal()
    return mtx_d + row_sums_excluding_diagonal(abs(mtx))
