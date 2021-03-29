#!/usr/bin/env python
# coding: utf-8

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bcert as bc
import numpy as np

class MaxFunction(bc.FunLipschitz):
    def __init__(self, dim=2):
        super().__init__(dim)
    def f(self, x):
        return np.cos(x[0])+np.sin(x[1])
    def bound_grad(self):
        return np.sqrt(2)

def find_maximum():
    print("Finding an upper bound on the maximum of the function f(x0,x1)=cos(x0)+sin(x1) for (x0,x1) in [-pi,pi]^2")
    fun = MaxFunction()
    space = bc.CompactSpace(np.array([[-np.pi,np.pi]]*2))
    step = np.pi/4
    tol = 1e-1
    res = bc.maxCert(fun, space, step, guess=2, tol=tol)
