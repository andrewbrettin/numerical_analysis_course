"""
steffensen.py

Implements Steffensen's method for finding the root of a function f.
"""

# Imports
import numpy as np
import matplotlib.pyplot

# Globals
f = lambda x : x**2 - 5
x_0 = 1.0
TOL = 1e-8
MAX_ITERATIONS = 1000

# Functions
def steffensen_iteration(f, x_k):
    """
    Given a function f and value x_k, returns x_{k+1}
    defined by steffensen's iteration.
    
    Parameters:
        f : callable
            function to find roots of.
        x_k : float
            Value to iterate.
    Returns:
        x_next
            Value of x_{k+1}.
    """
    denom = (f(x_k + f(x_k)) - f(x_k)) / f(x_k)
    x_next = x_k - f(x_k) / denom
    return x_next

def find_root(f, x_0, tol=TOL, max_iter=MAX_ITERATIONS):
    """
    Applies steffensen's method for finding a root.
    
    Parameters:
        f : callable
            function to find roots of.
        x_0 : float
            Initial value.
        tol : float
            Tolerance for defining convergence
        max_iter : int
            Number of iterations to evaluate before raising error.
            
    Returns:
        x_list : list of floats
            The iterates of the steffensen method.
    """
    # Initialization:
    i = 0
    x_list = [x_0]  # List of iterates
    error = np.inf    # Initialize error to be inf to ensure larger than tol
    
    while error > tol and i < max_iter:
        # Iteration
        x_k = x_list[-1]
        x_new = steffensen_iteration(f, x_k)
        
        # Update
        i = i + 1
        error = np.abs(x_new - x_k)
        x_list.append(x_new)
    
    x_list = np.array(x_list)
    return x_list

def main():
    x_list = find_root(f, x_0)
    print(x_list[-1])
    print(f"f(x) = {f(x)}")
    
if __name__ == "__main__":
    main()
