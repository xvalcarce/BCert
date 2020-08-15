# coding: utf-8

import os
import numpy as np

from multiprocessing import Pool
from .compact_space import CompactSpace

def boundCert(fun, space, lmbd, step, exclude_element=None, 
        thread=0, verbose=0, **kwargs):
    """ Certification of an upper bound of a function with a bounded gradient.

    Parameters
    ----------
    fun : bcert.FunLipschitz
        Function to certify the bound on.
    space : bcert.CompactSpace
        A compact space from the BCert class.
    step : int
        Distance between first neighbors grid element.
    exclude_elem : None or numpy.ndarray
        2D-array, each element are point to exclude from the grid. Default `None`.
    thread : int
        Corresponding thread when used from parallelization.
    verbose : int
        Verbose each `verbose` element of the discretized space.

    Returns
    -------
    f_max : int
        Maximum of the function evolution found.
    g_max : list
        Point for which `f_max` was obtain.
    """
    depth = kwargs.get("depth", 0)
    max_depth = kwargs.get("max_depth",-1)
    f_max = kwargs.get("f_max", -np.inf)
    g_max = kwargs.get("g_max", None)
    elem = kwargs.get("elem", None)
    grid = space.discretized(step) if depth == 0 else space.discr_element(elem, step/2)
    for e,g in enumerate(grid):
        if type(exclude_element) is np.ndarray:
            if np.array([(g-step/2<=_).all() and (g+step/2>=_).all() for _ in exclude_element]).any():
                continue
        if (verbose != 0) and (depth == 0) and not (e % verbose):
            print("{} : {}  {}".format(thread,e,g))
        eval_f = fun.f(g)
        if eval_f >= lmbd:
            raise Exception("Exceed or found {}: reach {} at grid element {}, certification failed.".format(lmbd, eval_f, g))
        evol_f = eval_f+(step*fun.b_grad*np.sqrt(fun.dim)/2)
        exceed_lmbd = evol_f > lmbd
        if exceed_lmbd:
            f_max_tmp, g_max_tmp = boundCert(fun, space, lmbd, step/2,
                    exclude_element=exclude_element, thread=thread, verbose=verbose, f_max=f_max, 
                    g_max=g_max, depth=depth+1, elem=g)
            if f_max_tmp > f_max:
                f_max = f_max_tmp
                g_max = g_max_tmp
        else:
            if evol_f >= f_max:
                f_max = evol_f
                g_max = g
    if depth == 0:
        print("Thread {} : maximum value found {} for x={}".format(thread,f_max,g_max))
        print("Certification sucessfull.")
    return f_max, g_max

def maxCert(fun, space, step, guess=-np.inf, tol=1e-2, exclude_element=None, 
        thread=0, verbose=0, **kwargs):
    """ Maximization of a Lipschitz function. Gives a certified with the solution
    being at most sqrt(dim)*tol/2 close to the maximum.

    Parameters
    ----------
    fun : bcert.FunLipschitz
        Lipschitz function to maximize.
    space : bcert.CompactSpace
        A compact space from the BCert class.
    step : int
        Distance between first neighbors grid element.
    guess : Float
        Guess (lower bound) on the maximum to find.
    exclude_elem : None or numpy.ndarray
        2D-array, each element are point to exclude from the grid. Default `None`.
    thread : int
        Corresponding thread when used from parallelization.
    verbose : int
        1: Verbose each `verbose` element of the discretized space. 2: Verbose
        each new maximum found respecting tol.

    Returns
    -------
    f_max : int
        Upper bound on the maximum of the function found.
    g_max : list
        Solution to `f_max`.
    """
    depth = kwargs.get("depth", 0)
    max_depth = kwargs.get("max_depth",-1)
    f_max = kwargs.get("f_max", -np.inf)
    g_max = kwargs.get("g_max", None)
    elem = kwargs.get("elem", None)
    grid = space.discretized(step) if depth == 0 else space.discr_element(elem, step/2)
    min_step = np.round(np.log2(step/tol))
    for e,g in enumerate(grid):
        if type(exclude_element) is np.ndarray:
            if np.array([(g==_).all() for _ in exclude_element]).any():
                continue
        if (verbose == 1) and (depth == 0) and not (e % verbose):
            print("{} : {}  {}".format(thread,e,g))
        eval_f = fun.f(g)
        evol_f = eval_f+(step*fun.b_grad*np.sqrt(fun.dim)/2)
        exceed_guess = evol_f >= guess
        if exceed_guess and step>tol:
            f_max_tmp, g_max_tmp = maxCert(fun, space, step/2, guess, tol=tol,
                    exclude_element=exclude_element, thread=thread, verbose=verbose, f_max=f_max, 
                    g_max=g_max, depth=depth+1, elem=g)
            if f_max_tmp >= f_max:
                f_max = f_max_tmp
                g_max = g_max_tmp
        elif exceed_guess:
            if evol_f > f_max and step<=tol:
                f_max = evol_f
                g_max = g
                if verbose == 2:
                    print("Found new max at {} around point {}, depth {} step {}".format(f_max,g_max, depth, step))
    if depth == 0:
        print("Optimization terminated:")
        print("\tMaximum value found f(x) = {}, with x = {}±{}".format(f_max,g_max,step/2**min_step))
    return f_max, g_max

def maxCertSingleLevel(g, fun, space, step, guess):
    """ Evaluation of one hypercube. Return a new_grid the f(k)>guess.
    """
    eval_f = fun.f(g)
    evol_f = eval_f+(step*fun.b_grad*np.sqrt(fun.dim)/2)
    exceed_guess = evol_f >= guess
    if exceed_guess:
        new_grid = space.discr_element(g, step/4)
        return new_grid
    else:
        return False

def maxCertSingleEval(g, fun, step, guess, ret_g=False):
    """ Single evaluation, without the grid generation.
    """
    eval_f = fun.f(g)
    evol_f = eval_f+(step*fun.b_grad*np.sqrt(fun.dim)/2)
    exceed_guess = evol_f > guess
    if exceed_guess:
        return np.concatenate(([evol_f],g)) if ret_g else evol_f
    else:
        return np.concatenate(([-np.inf],g)) if ret_g else -np.inf

def maxCertPar(fun, space, step, guess=-np.inf, tol=1e-2, threads=4):
    """ Parallelization of the `maxCert` function.

    Parameters
    ----------
    fun : bcert.FunLipschitz
        Function to certify the bou
    bound : numpy.ndarray
        Bound, 2D-array, to create a compact space from the BCert class.
    step : int
        Distance between first neighbors grid element.
    exclude_elem : None or numpy.ndarray
        2D-array, each element are point to exclude from the grid. Default `None`.
    threads : int
        Number of threads to use. Default is the number of core found on
        the device, or `1` if the preceding is inapplicable.

    Returns
    -------
    res : numpy.array
       Solution `f_max` and the corresponding variable.
    """
    min_step = np.round(np.log2(step/tol))
    grid = space.discretized(step)
    while step>tol:
        with Pool(processes=threads) as pool:
            res = pool.starmap(maxCertSingleLevel, [(g,fun,space,step,guess) for g in grid])
        grid = np.concatenate([r for r in res if r is not False])
        step /= 2
    with Pool(processes=threads) as pool:
        res = pool.starmap(maxCertSingleEval, [(g,fun,step,guess,True) for g in grid])
    res = np.array(res)
    res = res[res[:,0].argsort()]
    sol = res[-1]
    print("Optimization terminated:")
    print("\tMaximum value found f(x) = {}, with x = {}±{}".format(sol[0],sol[1:],step/2**min_step))
    return res[-1]

def boundCertPar(fun, bound, lmbd, step, exclude_element=None, 
        threads=None, verbose=0):
    """ Parallelization of the `boundCert` function. This will subvide in `threads` the domain,
    this in the direction with the greatest distance between boundaries.

    Parameters
    ----------
    fun : bcert.FunLipschitz
        Function to certify the bound on.
    bound : numpy.ndarray
        Bound, 2D-array, to create a compact space from the BCert class.
    step : int
        Distance between first neighbors grid element.
    exclude_elem : None or numpy.ndarray
        2D-array, each element are point to exclude from the grid. Default `None`.
    threads : int
        Number of threads to use. Default is the number of core found on
        the device, or `1` if the preceding is inapplicable.
    verbose : int
        Verbose each `verbose` element of the discretized space.

    Returns
    -------
    res : list
        List of output of `boundCert` for every subdomain.

    """
    if threads is None:
        try:
            threads = os.cpu_count()
        except:
            threads = 1
            print("Couldn't use automatic number of thread detection.",
            "This might occure when using virtualization.")
    bound_max = np.argmax([b[1]-b[0] for b in bound])
    bound_subdiv = bound[bound_max][1]/threads
    bound_div = [bound[:bound_max].tolist() + [[bound_subdiv*_, bound_subdiv*(_+1)]] 
            + bound[bound_max+1:].tolist() for _ in range(threads)]
    if step > bound_subdiv:
        step = bound_subdiv/2
        print("Step larger than bound_subdiv, changing it's value to {}".format(step))
    spaces = [CompactSpace(np.array(b)) for b in bound_div]
    with Pool(processes=threads) as pool:
        res = pool.starmap(boundCert, [(fun, s, lmbd, step,
            exclude_element,t,verbose) for t,s in enumerate(spaces)])
        print("Found max:\n",res)
        print("Certified.\n")
        return res

def boundCertUserPar(fun, bounds, lmbd, step, exclude_element=None,
        verbose=0):
    """ Parallelization of the `boundCert` function. One thread per provided bound.

    Parameters
    ----------
    fun : bcert.FunLipschitz
        Function to certify the bound on.
    bounds : numpy.ndarray or list
        List of bounds -- 3D-array -- to create compact spaces from each
        of the bound from the BCert class.
    step : int
        Distance between first neighbors grid element.
    exclude_elem : None or numpy.ndarray
        2D-array, each element are point to exclude from the grid. Default `None`.
    verbose : int
        Verbose each `verbose` element of the discretized space.

    Returns
    -------
    res : list
        List of output of `boundCert` for every subdomain.
    """
    threads = len(bounds)
    spaces = [CompactSpace(b) for b in bounds]
    with Pool(processes=threads) as pool:
        res = pool.starmap(boundCert, [(fun, space, lmbd, step,
            exclude_element,t,verbose) for t,space in enumerate(spaces)])
        print("Found max:\n",res)
        print("Certified.\n")
        return res
