# coding: utf-8

import numpy as np

from abc import abstractmethod

class FunLipschitz(object):
    """ Abstract class representing function witch bound on the gradient.

    """
    def __init__(self, dim, fix_param=None, orientation_grad=None):
        self.dim = dim
        self.fix_param = fix_param
        self.b_grad = self.bound_grad()
        self.orientation_grad = np.zeros(dim) if (orientation_grad is 
                None) else orientation_grad

    @abstractmethod
    def f(self,x):
        """ The function itself. Please implement it.

        Parameters
        ----------
        x : list
            List of input of the function. Should be the same size as `dim`.

        Returns
        -------
        y : float
            Scalar corresponding to `f(x)`.
        """
        raise NotImplemented("Please implement the function")

    @abstractmethod
    def bound_grad(self):
        """ Bound on the gradient of f(x).

        Returns
        -------
            : float
            Scalar corresponding to an upper bound on the gradient of `f`.
        """
        raise NotImplemented("Please implement the function: return a scalar bound of the norm of the gradient")
