import numpy as np
import scipy as sc
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

from scipy.linalg import svd, norm

CMAP = plt.get_cmap('RdBu_r')
CMAP.set_bad('k')

class EOF_model():
    def __init__(self, data, weights=None):
        """
        Creates the EOF model
        
        Parameters:
            data : xarray.DataArray
                Data to perform EOF analysis on
            weights: xarray.DataArray
                Weights to give to gridpoints (e.g., gridcell area)
        """
        if isinstance(data, xr.DataArray):
            self.darray = data
        else:
            raise TypeError(
                "Data should be type xarray.DataArray. That's sort of \
                the whole point"
            )
        if weights is not None:
            self.darray = self.darray * np.sqrt(weights)
        
    def fit(self, pc_dim='time'):
        """
        Performs an EOF analysis of a dataset. Takes a PC dimension name
        as an argument. All other dimensions are stacked into eof_dims.
        
        Parameters:
        
        Initializes:
            df : pandas.DataFrame
            PCs : xarray.DataArray
            EOFs : xarray.DataArray
            SVs : xarray.DataArray
        """
        if pc_dim not in self.darray.dims:
            raise ValueError('{} not in darray.dims'.format(pc_dim))
        self.pc_dim = pc_dim
        self.eof_dims = tuple(
            [dim for dim in self.darray.dims if dim != self.pc_dim]
        )
        
        # Take mean in pc_dim
        self.darray = self.darray - self.darray.mean(dim=pc_dim)
        
        # Populate table
        self.df = (
            self.darray
            .stack(multiindex=self.eof_dims)
            .to_pandas()
            .dropna(axis=1)
        )
        
        X = self.df.values
        N, M = np.shape(self.df)
        
        U, S, Vh = svd(X)
        
        # Principal components
        self.PCs = xr.DataArray(U, 
            dims=(pc_dim, 'n'),
            coords = {
                'time' : self.df.index,
                'n' : np.arange(1, N+1)
            }
        )
        
        # Singular values
        self.SVs = xr.DataArray(
            S, dims=('n',),
            coords = {'n' : np.arange(1, len(S) + 1)}
        )
        
        # Empirical Orthogonal Functions
        V = Vh.transpose()
        V = pd.DataFrame(
            V, index=self.df.columns, columns=np.arange(1, M+1)
        )
        
        self.EOFs = pd.DataFrame(
            V, index=self.df.columns, columns=np.arange(1, M+1)
        ).stack()
        
        self.EOFs.index.names = [*self.eof_dims, 'n']
        self.EOFs = self.EOFs.to_xarray()
        
        for dim in self.eof_dims:
            self.EOFs = self.EOFs.sortby(dim)
        
    def get_PC(self, k):
        """
        Returns kth PC.
        
        Parameters:
            k : int
                Principal component to select.
        Returns:
            PC : xarray.DataArray
                kth principal component.
        """
        PC = self.PCs.sel(n=k)
        return PC
    
    def get_EOF(self, k):
        """
        Returns kth EOF.
        
        Parameters:
            k : int
                EOF to select.
        Returns:
            EOF : xarray.DataArray
                kth principal component.
        """
        EOF = self.EOFs.sel(n=k)
        return EOF
    
    def get_SV(self, k):
        """
        Returns kth singular value.
        
        Parameters:
            k : int
                EOF to select.
        Returns:
            sigma : float
                kth principal component.
        """
        sigma = self.SVs.sel(n=k).item()
        return sigma
    
    def variances(self):
        """
        Returns the variances explained by each principal component.
        
        Returns:
            variances : xarray.DataArray
                Variances of each PC.
        """
        variances = self.SVs**2 / norm(self.SVs)**2
        return variances
        
    def cummulative_variance(self, r):
        """
        Computes the variance explained by the rank r approximation.
        
        Parameters:
            r : int
                Rank of approximation.
        Returns :
            var : float
                Variance explained by rank r approximation.
        """
        var = sum(self.variances()[0:r]).item()
        return var
    
    def min_rank(self, var=0.9):
        """
        Given a net variance, selects the minimum rank r needed to obtain 
        that much variance.
        
        Parameters:
            var : float
                Total variance required
        Returns:
            r : int
                Minimum rank to obtain variance var.
        """
        if not 0 <= var <= 1:
            raise ValueError('var must take values between 0 and 1')
        r = 0
        cum_var = 0
        while cum_var < var:
            r = r + 1
            cum_var = self.cummulative_variance(r)
        return r
    
    def low_rank_approximation(self, r=10, net_variance=None):
        """
        Computes the rank r approximation to the dataset using
        the first r principal components.
        
        If r is specified, then
        
        Otherwise, if net_variance is specified, 
        
        Parameters:
            r : int
                Rank of approximation.
            net_variance : float, 
        """
        
        if isinstance(r, int):
            approx = sum([
                (self.SVs.sel(n=k) * self.EOFs.sel(n=k) * self.PCs.sel(n=k))
                .drop('n') for k in range(1, r+1)
            ])
            return approx
        elif isinstance(net_variance, float):
            r = self.min_rank(var=net_variance)
            approx = sum([
                (self.SVs.sel(n=k) * self.EOFs.sel(n=k) * self.PCs.sel(n=k))
                .drop('n') for k in range(1, r+1)
            ])
            return approx
        else:
            raise ValueError('Rank or variance not specified correctly.')
    
    def plot_PC(self, k, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        PC = self.get_PC(k)
        PC.plot(ax=ax)
        
        ylim = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim(-ylim, ylim)

        ax.set_title('PC {}'.format(k))
        
    def plot_EOF(self, k, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        EOF = self.get_EOF(k)
        EOF.plot(ax=ax, cmap=CMAP)
        ax.set_title('EOF {}'.format(k))
        
    def plot_variances(self, kmax=10, cumulative=False, ax=None):
        """
        Plots the variances of 
        """
        if ax is None:
            fig, ax = plt.subplots()
            
        ks = np.arange(1, kmax+1)
        
        if cumulative:
            sum_var = [
                np.sum(self.variances()[0:k]) for k in ks
            ]
            ax.plot(ks, sum_var, ls='-', marker='x')
            ax.set(
                title='Singular values (normalized)',
                xlabel=r'Index $i$',
                ylabel=r'Cumulative variance of PC',
                ylim=(0, 1),
                xticks=ks
            );
        else:
            ax.plot(
                ks, self.variances()[0:kmax], 
                ls='-', marker='x'
            )
            ax.set(
                title='Singular values (normalized)',
                xlabel=r'Index $i$',
                ylabel=r'Variance of PC',
                ylim=(0, 1),
                xticks=ks
            );