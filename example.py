#!/usr/bin/env python
# coding: utf-8

import bcert as bc
import numpy as np

class Dummy(bc.FunLipschitz):
    def __init__(self):
        super().__init__()

    def f(self, x):
        return np.cos(x[0])+np.sin(x[1])

    def bound_df(self):
        bound_dx0 = 1
        bound_dx1 = 1
        return np.array([bound_dx0,bound_dx1])

if __name__ == "__main__":
    print("Certifying the function f(x0,x1)=cos(x0)+sin(x1) < 2 for (x0,x1) in ([0,2pi],[0,pi])\({0,pi/2},{2pi,pi/2})")
    fun = Dummy()
    space = bc.CompactSpace(np.array([[0,2*np.pi],[0,np.pi/2]]))
    lmbd = 2
    step = np.pi/10
    exclude = np.array([[0,np.pi/2],[2*np.pi,np.pi/2]])
    
    try:
        f_max, x_max = bc.boundCert(fun, space, lmbd, step)
    except:
        print("Failed.")
