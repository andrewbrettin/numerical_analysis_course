"""
lu_utils
    Various utility functions for LU factorization.
    
"""

import numpy as np


def lu(A):
    """
    Given an nxn matrix A, computes the LU decomposition without
    pivoting.
    
    Parameters:
        A : np.ndarray
    Returns
        L, U : np.ndarray
    """
    # Check shapes
    if np.shape(A)[0] != np.shape(A)[1]:
        raise ValueError("Matrix dimensions should be the same length")
    
    n = np.shape(A)[0]
    L = np.eye(n)
    U = np.zeros((n,n))
    
    # Set first row
    U[0, :] = A[0, :]
    
    for i in np.arange(2, n+1):
        for j in np.arange(1, i):
            # L_ij factors
            sum_prod = np.sum(
                [L[i-1, k-1]*U[k-1, j-1] for k in np.arange(1, j)]
            )
            L[i-1, j-1] = 1/U[j-1, j-1] * (A[i-1, j-1] - sum_prod)
            
        for j in np.arange(i, n+1):
            # U_ij factors
            sum_prod = np.sum(
                [L[i-1, k-1]*U[k-1, j-1] for k in np.arange(1, i)]
            )
            U[i-1, j-1] = A[i-1, j-1] - sum_prod
    
    return L, U