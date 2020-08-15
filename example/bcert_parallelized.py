#!/usr/bin/env python
# coding: utf-8

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bcert as bc
import numpy as np

class SimpleFunction(bc.FunLipschitz):
    def __init__(self, dim=2):
        super().__init__(dim)
    def f(self, x):
        return np.cos(x[0])+np.sin(x[1])
    def bound_grad(self):
        return np.sqrt(2)

def certification():
    print("Certifying the function f(x0,x1)=cos(x0)+sin(x1) < 2 for (x0,x1) in ([0,2pi],[0,pi])\({0,pi/2},{2pi,pi/2})")
    fun = SimpleFunction()
    bound = np.array([[0,np.pi/2],[0,np.pi/2]])
    lmbd = 2
    step = np.pi/32
    exclude = np.array([[0,np.pi/2]])

    print("Default multithreading") 
    try:
        res = bc.boundCertPar(fun, bound, lmbd, step, exclude_element=exclude)
    except:
        print("Certification failed with default parallelization method.")

if __name__ == "__main__":
    certification()
