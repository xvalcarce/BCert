#!/usr/bin/env python
# coding: utf-8

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bcert as bc
import numpy as np
from itertools import product

#--- State distribution and violation param ---#

nu = .061
pf = .61381508
q = .5
meta = (nu,pf,q)

#--- Fidelity inheriting from FunLipschitz ---#

class fidelity_an(bc.FunLipschitz):
    def __init__(self, dim=5):
        super().__init__(dim)

    def f(self, x, meta=meta):
        """ Compute the fidelity with the parametrisation using angles """
        a0,a1,b0,b1,theta = x
        nu,pf,q = meta
        da,db = [[np.cos(s0)*np.cos(s1)+np.sin(s0)*np.sin(s1),np.cos(s0)*np.cos(s1)-np.sin(s0)*np.sin(s1),-1+np.cos(s0)**2+np.cos(s1)**2] for s0,s1 in [(a0,a1),(b0,b1)]]
        a1,b1 = [s0**2-s1**2 for s0,s1 in [(np.cos(a0),np.cos(a1)),(np.cos(b0),np.cos(b1))]]
        eps = nu*(a1*b1 + da[0]*db[0] + np.abs(da[1]*db[1]+da[2]*db[2])) \
            + (1-nu)*pf*np.sqrt((q+(1-q)*np.cos(theta))**2+(1-q)**2 * np.sin(theta)**2) \
            + (1-nu)*(1-pf)*(np.cos(theta)*b1+np.sin(theta)*db[0])
        return eps

    def bound_grad(self, meta=meta):
        nu,pf,q = meta
        b_a0 = 3*nu
        b_a1 = 3*nu
        b_b0 = 3*nu + (1-pf)*(1-nu)
        b_b1 = 3*nu + (pf-1)*(nu-1)
        b_theta = (pf-1)*(nu-1)
        grad = np.array([b_a0,b_a1,b_b0,b_b1,b_theta])
        norm_grad = np.linalg.norm(grad)
        return norm_grad

#--- Certification of a trivial fidelity ---#

def certification_threshold():
    fid = fidelity_an()
    lmbd = 1.
    div = [[0,np.pi/8],[np.pi/8,3*np.pi/8],[3*np.pi/8,np.pi/2]]
    bounds = [[i]+[j]+[k]+[[0,np.pi/2]]+[[0,np.pi/2]] for i,j,k in product(div, repeat=3)]
    step = np.pi/8
    hard = [3,4,6,7,12,13,15,16]

    print("Certifying blocks with low fidelity, starting with a \
            high step of {}".format(step))
    triv_hypc = np.array([b for i,b in enumerate(bounds) if i not in hard])
    try:
        res0 = bc.boundCertUserPar(fid, triv_hypc, lmbd, 
                step, verbose=100)
    except:
        print("Certification on the trivial blocks failed.")

    print("Focusing on remaining blocks.")
    print("Subdividing remaining blocks in 9 hypercubes.")
    rem_hypc = np.array([[bounds[i][:3]+[d0]+[d1] for d0,d1 in \
            product(div, repeat=2)] for i in hard])
    triv_rem_hypc = rem_hypc[:,np.ix_([0,1,2,5,8])].reshape(40,5,2)
    step = np.pi/16
    print("Certificatying sub-blocks 0,1,2,5 and 8 \
            with a smaller step {}".format(step))
    try:
        res1 = bc.boundCertUserPar(fid, triv_rem_hypc, lmbd, 
                step, verbose=100)
    except:
        print("Certification on the trivial sub-blocks failed.")

    step = np.pi/96
    print("Certifying the lasts non-trivial sub-blocks with a \
            smaller step of {}".format(step))
    ntriv_rem_hypc = rem_hypc[:,np.ix_([3,4,6,7])].reshape(32,5,2)
    # We remove the block around 0,pi/2,0,pi/2,0 with has fid = 1
    ntriv_rem_hypc = np.delete(ntriv_rem_hypc,10,0)
    try:
        res2 = bc.boundCertUserPar(fid, ntriv_rem_hypc, lmbd, 
                step, verbose=100)
    except:
        print("Certification on the non-trivial sub-blocks failed.")

#--- Run as script ---#

if __name__ == "__main__":
    certification_threshold()
