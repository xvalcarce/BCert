#!/usr/bin/env python
# coding: utf-8

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
import numpy as np
import bcert as bc
from itertools import product
from multiprocessing import Pool

# Param:
# beta, gamma -> 0,pi/4 | alpha, phi -> 0,pi/2
# L1 = cos(alpha)cos(beta)
# L2 = cos(alpha)sin(beta)
# L3 = sin(alpha)cos(gamma)
# L4 = sin(alpha)sin(gamma)
# phi -> angle mesure Alice

class FunCert(bc.FunLipschitz):
    def __init__(self, omega, q, t, dim=4):
        self.omega = omega
        self.q = q
        self.sqrtq = np.sqrt(q)
        self.t = t
        self.co2 = np.cos(self.omega)**2
        self.so2 = np.sin(self.omega)**2
        super().__init__(dim)
    def f(self,x):
        #Attributing Variable
        sqrtL1 = np.cos(x[0])*np.cos(x[1])
        sqrtL2 = np.cos(x[0])*np.sin(x[1])
        sqrtL3 = np.sin(x[0])*np.cos(x[2])
        sqrtL4 = np.sin(x[0])*np.sin(x[2])
        L1 = sqrtL1**2
        L2 = sqrtL2**2
        L3 = sqrtL3**2
        L4 = sqrtL4**2
        cp = np.cos(x[3])
        sp = np.sin(x[3])
        cp2 = cp**2
        sp2 = sp**2
        #Avoid repetition
        r11 = cp*sqrtL1*sqrtL3*self.sqrtq
        r12 = sp*sqrtL1*sqrtL4*self.sqrtq
        r21 = sp*sqrtL2*sqrtL3*self.sqrtq
        r22 = -cp*sqrtL2*sqrtL4*self.sqrtq
        #Conditional state
        rho_c = np.array([[L1,0,r11,r12],[0,L2,r21,r22],[r11,r21,L3,0],[r12,r22,0,L4]])
        #State
        rho = np.diag([L1,L2,L3,L4])
        #Beta
        Tz = L1-L2+L3-L4
        Tx = L1-L2-L3+L4
        Tz2 = Tz**2
        Tx2 = Tx**2
        B = (1/np.sqrt(2))*np.sqrt(self.co2*(Tz2*cp2+Tx2*sp2)+self.so2*(Tx2+Tz2)+
                np.sqrt((self.co2*(Tz2*cp2-Tx2*sp2)-self.so2*(Tz2-Tx2))**2
                    +4*(self.co2*cp*sp*Tz*Tx)**2))
        H_rho = -np.sum([l*np.log(l) for l in [L1,L2,L3,L4]])
        lambdas = np.abs(np.linalg.eigvals(rho_c))
        H_rho_c = -np.sum([np.log(l**l) for l in lambdas])
        obj = H_rho - H_rho_c + self.t*B
        return obj.real

    def minusf(self,x):
        return -self.f(x)

    def bound_grad(self):
        lc = 12.7+7*self.t
        return lc

def mcgrid(fun,grid,step,guess,thread):
    save_g = []
    for g in grid:
        r = bc.bcert.maxCertSingleEval(g,fun,step,guess,ret_g=True)
        if r[0] != -np.inf:
            save_g.append(r)
    save_g = np.array(save_g)
    print("Done with thread {}:\n\tEleminated {}/{} element.".format(thread,grid.shape[0]-save_g.shape[0],grid.shape[0]))
    return save_g

def mc(fun,space,step,guess,thread):
    save_g = []
    grid = space.discretized(step)
    for g in grid:
        r = bc.bcert.maxCertSingleEval(g,fun,step,guess,ret_g=True)
        if r[0] != -np.inf:
            save_g.append(r)
    save_g = np.array(save_g)
    print("Done with thread {}:\n\tEleminated {}/{} element.".format(thread,grid.shape[0]-save_g.shape[0],grid.shape[0]))
    return save_g

def certification(threads=24,tol=0.001):
    guess = 5.4459020047
    omega = 0.9580682478 
    fun = FunCert(omega,1.,guess)
    s = (np.pi/2)/threads
    step = s/2
    spaces = [bc.CompactSpace(np.array([[i*s,(i+1)*s],[0,np.pi/4],[0,np.pi/4],[0,np.pi/2]])) for i in range(threads)]
    start = time.time()
    with Pool(processes=24) as pool:
        r = pool.starmap(mc,[(fun,space,step,5.455195177,t) for t,space in enumerate(spaces)])
    rr = np.concatenate([_ for _ in r if _.size>0])
    stop = time.time()-start
    print("(Took {}) Upper bound on max found for step {} with boundgrad {} is: {}".format(stop,step,fun.b_grad*step*np.sqrt(4)/2,np.max(rr[:,0])))
    np.savetxt("./data/rr_step_{}.csv".format(step),rr,delimiter=',')
    step /= 2
    while step>tol:
        new_center = (step/2)*np.array([c for c in product(range(-1,2,2), repeat=4)])
        gr = np.array([new_center+r for r in rr[:,1:]])
        gr = gr.reshape(gr.shape[0]*16,4)
        div = gr.shape[0]/threads
        gr = np.array([gr[round(div*i):round(div*(i+1))] for i in range(threads)],dtype=object)
        start = time.time()
        with Pool(processes=threads) as pool:
            r = pool.starmap(mcgrid,[(fun,g,step,guess,t) for t,g in enumerate(gr)])
        rr = np.concatenate([_ for _ in r if _.size>0])
        stop = time.time()-start
        print("(Took {}) Max found for step {} with boundgrad {} is: {}".format(stop,step,fun.b_grad*step*np.sqrt(4)/2,np.max(rr[:,0])))
        np.savetxt("./data/rr_step_{}.csv".format(step),rr,delimiter=',')
        step /= 2
        
if __name__ == "__main__":
    certification()
