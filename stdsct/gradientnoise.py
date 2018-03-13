"""
Gradient Descent with noise
"""

import warnings
import numpy as np
import numpy.random as rd
import os
import sys
from scipy.optimize import *


__all__ = ['GradientNoise']


def GradientNoise(Func, Jac, gamma, amp, tol, x_init, itmax, K, it_init=None,
                  bounds=None, DirDoF=None):
    """
    Gradient Descent with Noise Algorithm
    PARAMETERS:
    -----------
       * Func : callable
                Function to minimize
       * Jac  : callable
                Jacobian of the function to minimize
       * gamma : float
                Real number that divide the gradient
       * amp :  float
                Real number, amplitude of the noise
       * tol :  flaot
                Real number, tolerance for the convergence
       * x_init  : ndarray
                Initial guess of x
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
       * min_Func : flaot
            value of the minimum of Func(x)
       * min_it : int
            iteration number where the algorithm stopped
    """
    print "Gradient Descent Simulated Annealing..."
    N = len(x_init)
    min_x = np.copy(x_init)
    min_Func = Func(x_init)
    Energy = min_Func

    if it_init is None:
        it_init = 0

    min_it = it_init

    # Save the DOF
    if DirDoF is not None:
        SavedDOF = x[DirDoF]

    print "Initial Gradient = ", np.linalg.norm(Jac(x_init)), \
          "Initial Energy = ", Energy
    for j in range(K):
        it = it_init
        res = Func(x_init)
        x = np.copy(x_init)
        while (it < itmax):
            dt = 1.0/np.log(float(it+2))**0.5
            sigma = amp*np.sqrt(gamma/(np.log(it+2)*dt))*1.0/float(it+2)
            b = Jac(x)
            new_x = x - dt/gamma*b + \
                sigma*np.sqrt(dt)/gamma*rd.normal(0.0, 1.0, x.shape)

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
            min_it = it

    return [min_x, min_Func, min_it]
