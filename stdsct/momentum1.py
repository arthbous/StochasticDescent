"""
* Momentum 1 with noise
"""

import warnings
import numpy as np
import numpy.random as rd
import os
import sys
from scipy.optimize import *


__all__ = ['Momentum1']


def Momentum1(Func, Jac, lda, gamma, amp, tol, x_init, v_init,
              itmax, K, it_init=None, bounds=None, DirDoF=None):
    """
    Momentum 1 Algorithm
    PARAMETERS:
    -----------
       * Func : callable
                Function to minimize
       * Jac  : callable
                Jacobian of the function to minimize
       * lda : float
                value is between 0 and 1
       * gamma : float
                Real number that divide the gradient
       * amp :  float
                Real number, amplitude of the noise
       * tol :  flaot
                Real number, tolerance for the convergence
       * x_init  : ndarray
                Initial guess of x
       * v_init  : ndarray
                Initial guess for Momentum 1 (velocity), same shape as x0
       * itmax  : int
                maximum number of iteration
       * K  : int
                number of times to repeat the algorithm
       * it_init : int, optional
                iteration number where to start the algorithm
       * bounds : sequence
                 dimension 2 where each value of x has to be between
       * DirDoF : list of tuple
                degree(s) of freedom where x will not change
     RETURNS:
     -------
       * x  : ndarray
            value of x where Func(x) is the minimum, same size of x_init
       * v  : ndarray
            value of Momentum 1 (velocity)
       * min_Func : flaot
            value of the minimum of Func(x)
       * min_it : int
            iteration number where the algorithm stopped
    """
    print "Momentum 1..."
    N = len(x_init)
    min_x = np.copy(x_init)
    min_v = np.copy(v_init)
    min_Func = Func(x_init)
    Energy = min_Func

    if it_init is None:
        it_init = 0

    min_it = it_init

    # Save the DOF
    if DirDoF is not None:
        SavedDOF = x[DirDoF]

    print "Initial Gradient = ", np.linalg.norm(Jac(x_init)),\
          "Initial Energy = ", min_Func
    for j in range(K):
        it = it_init
        res = Func(x_init)
        x = np.copy(x_init)
        v = np.copy(v_init)
        while (it < itmax):
            dt = 1.0/np.log(float(it+2))**0.5
            sigma = amp*np.sqrt(gamma/(np.log(it+2)*dt))*1.0/float(it+2)
            b = Jac(x)
            v = (1.0 - lda*dt)*v - dt*b/gamma + \
                sigma*np.sqrt(dt)/gamma*rd.normal(0.0, 1.0, x.shape)
            new_x = x + v*dt

            if (bounds is not None) and (len(bounds) == 2):
                choice = [new_x > bounds[0], new_x < bounds[1]]
                x = np.select(choice, [new_x, x])
            else:
                x = new_x
            if DirDoF is not None:
                x[DirDoF] = SavedDOF

            it += 1
            PrevEnergy = Energy
            Energy = Func(x)
            if abs(PrevEnergy-Energy)/abs(Energy) < tol:
                break

        res = np.linalg.norm(Jac(x))
        Energy = Func(x)
        print "\t At iteration ", it, ", Gradient = ", res, "Energy = ", Energy
        if Energy < min_Func:
            min_Func = Energy
            min_x = np.copy(x)
            min_v = np.copy(v)
            min_it = it

    return [min_x, min_v, min_Func, min_it]
