#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:44:15 2023

@author: archie
"""

import numpy as np


# %% Diagonal preconditioners
def row_sums_excluding_diagonal(mtx):
    # Set the diagonal elements to zero in the CSR matrix
    mtx = mtx.copy()
    mtx.setdiag(0)

    # Compute the sum of row elements excluding the diagonal elements
    return np.array(mtx.sum(axis=1)).flatten()

def unit(x):
    x[x == 0] = 1
    return x / np.abs(x)

def diagp0(mtx):
    return mtx.diagonal()

def diagp1(mtx):
    return np.multiply(unit(diagp0(mtx)), np.maximum(np.abs(diagp0(mtx)), row_sums_excluding_diagonal(abs(mtx))))

def diagp2(mtx):
    return np.multiply(unit(diagp0(mtx)), np.array(abs(mtx).sum(axis=1)).flatten())

def diagl1(mtx):
    return diagp0(mtx) + row_sums_excluding_diagonal(abs(mtx))