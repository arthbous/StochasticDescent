#! /usr/bin/python2.7

"""
Different Descent Algorithms
 * Gradient Descent with noise
 * Moment 1 with noise
 * Moment 2 with noise
"""

import warnings
import numpy as np
import numpy.random as rd
import os
import sys
from scipy.optimize import *

# Moment 2 Algorithm
# INPUT:
#   * Func = Function to minimize
#   * Jac  = Jacobian of the function to minimize
#   * lda = vector of dimension 3 where each value is  between 0 and 1
#   * gamma =  Real number that divide the gradient
#   * amp=  Real number, amplitude of the noise
#   * tol =  Real number, tolerance for the convergence
#   * x_init  = Initial guess of x
#   * v_init  = Initial guess for moment 1 (velocity)
#   * z_init  = Initial guess for moment 2
#   * itmax  = Interger, maximum number of iteration
#   * K  = Interger, number of times to repeat the algorithm
#   * it_init (optional) = Integer, iteration number where to start the algorithm
#   * bounds (optional) = Real of dimension 2 where each value of x has to be between
#   * DirDoF (optional) = vector, degree(s) of freedom where x will not change
# OUPUT:
#   * x  = value of x where Func(x) is the minimum
#   * v  = value of moment 1 (velocity)
#   * z = value ofmoment 2
#   * min_Func = value of the minimum of Func(x)
#   * min_it = iteration number where the algorithm stopped
def Moment2(Func,Jac,lda,gamma,amp,tol,x_init,v_init,z_init, \
            itmax,K,it_init=None,bounds=None,DirDoF=None):
    print "Moment 2..."
    N = len(x_init)
    min_x = np.copy(x_init)
    min_v = np.copy(v_init)
    min_z = np.copy(z_init)
    min_Func = Func(x_init)
    Energy = min_Func
    mu = lda[0]
    B = lda[1]
    C = lda[2]

    if it_init==None:
        it_init=0

    min_it=it_init

    DoF = range(N)
    if DirDoF!=None:
        for i in DirDoF:
            DoF.remove(i)

    print "\t Initial Gradient = ", np.linalg.norm(Jac(x_init)), "Initial Energy = ", min_Func
    for j in range(K):
        it = it_init
        res = Func(x_init)
        x = np.copy(x_init)
        v = np.copy(v_init)
        z = np.copy(z_init)
        while ( (it < itmax) ):
            dt = 1.0/np.log(float(it+2))**0.5
            sigma = amp*np.sqrt(gamma/(np.log(it+2)*dt))*1.0/float(it+2)
            z = (1.0 -B*dt)*z + C*v*dt + sigma*rd.normal(0.0,1.0,N)*np.sqrt(dt)
            b = Jac(x)
            v = (1.0-mu*dt)*v - dt*b/gamma - z*dt
            new_x = x + v*dt

            if (bounds!=None and len(bounds)==2 ):
                for i in DoF:
                    if (new_x[i]>bounds[0] and new_x[i]<bounds[1]):
                        # print i, it, new_x[i], b[i], z[i], v[i], sigma, dt
                        x[i]=new_x[i]
            else:
                x = np.copy(new_x)

            it+=1
            PrevEnergy = Energy
            Energy = Func(x)
            if abs(PrevEnergy-Energy)/abs(Energy) < tol:
                break

        res = np.linalg.norm(Jac(x))
        Energy = Func(x)
        print "\t At iteration ",it,", Gradient = ",res, "Energy = ", Energy
        if (Energy < min_Func):
            min_Func = Energy
            min_x = np.copy(x)
            min_v = np.copy(v)
            min_z = np.copy(z)
            min_it=it

    print "After ",it,"iteration: Gradient = ",res, "Energy = ", min_Func

    return [min_x,min_v,min_z,min_Func,min_it]

# Moment 1 Algorithm
# INPUT:
#   * Func = Function to minimize
#   * Jac  = Jacobian of the function to minimize
#   * lda = real number between 0 and 1
#   * gamma =  Real number that divide the gradient
#   * amp=  Real number, amplitude of the noise
#   * tol =  Real number, tolerance for the convergence
#   * x_init  = Initial guess of x
#   * v_init  = Initial guess for moment 1 (velocity)
#   * itmax  = Interger, maximum number of iteration
#   * K  = Interger, number of times to repeat the algorithm
#   * it_init (optional) = Integer, iteration number where to start the algorithm
#   * bounds (optional) = Real of dimension 2 where each value of x has to be between
#   * DirDoF (optional) = vector, degree(s) of freedom where x will not change
# OUPUT:
#   * x  = value of x where Func(x) is the minimum
#   * v  = value of moment 1 (velocity)
#   * min_Func = value of the minimum of Func(x)
#   * min_it = iteration number where the algorithm stopped
def Moment1(Func,Jac,lda,gamma,amp,tol,x_init,v_init,\
        itmax,K,it_init=None,bounds=None,DirDoF=None):

    print "Moment 1..."
    N = len(x_init)
    min_x = np.copy(x_init)
    min_v = np.copy(v_init)
    min_Func = Func(x_init)
    Energy = min_Func

    if it_init==None:
        it_init=0

    min_it=it_init

    DoF = range(N)
    if DirDoF!=None:
        for i in DirDoF:
            DoF.remove(i)

    print "Initial Gradient = ", np.linalg.norm(Jac(x_init)), "Initial Energy = ", min_Func
    for j in range(K):
        it = it_init
        res = Func(x_init)
        x = np.copy(x_init)
        v = np.copy(v_init)
        while ( (it < itmax) ):
            dt = 1.0/np.log(float(it+2))**0.5
            sigma = amp*np.sqrt(gamma/(np.log(it+2)*dt))*1.0/float(it+2)
            b = Jac(x)
            v = (1.0-lda*dt)*v - dt*b/gamma + sigma*np.sqrt(dt)/gamma*rd.normal(0.0,1.0,N)
            new_x = x + v*dt

            if (bounds!=None and len(bounds)==2 ):
                for i in DoF:
                    if (new_x[i]>bounds[0] and new_x[i]<bounds[1]):
                        x[i]=new_x[i]
            else:
                x = np.copy(new_x)

            it+=1
            PrevEnergy = Energy
            Energy = Func(x)
            if abs(PrevEnergy-Energy)/abs(Energy) < tol:
                break

        res = np.linalg.norm(Jac(x))
        Energy = Func(x)
        print "\t At iteration ",it,", Gradient = ",res, "Energy = ", Energy
        if Energy < min_Func:
            min_Func = Energy
            min_x = np.copy(x)
            min_v = np.copy(v)
            min_it=it

    return [min_x,min_v,min_Func,min_it]


# Gradient descent with Noise Algorithm
# INPUT:
#   * Func = Function to minimize
#   * Jac  = Jacobian of the function to minimize
#   * gamma =  Real number that divide the gradient
#   * amp=  Real number, amplitude of the noise
#   * tol =  Real number, tolerance for the convergence
#   * x_init  = Initial guess of x
#   * itmax  = Interger, maximum number of iteration
#   * K  = Interger, number of times to repeat the algorithm
#   * it_init (optional) = Integer, iteration number where to start the algorithm
#   * bounds (optional) = Real of dimension 2 where each value of x has to be between
#   * DirDoF (optional) = vector, degree(s) of freedom where x will not change
# OUPUT:
#   * x  = value of x where Func(x) is the minimum
#   * min_Func = value of the minimum of Func(x)
#   * min_it = iteration number where the algorithm stopped
def GradientNoise(Func,Jac,gamma,amp,tol,x_init,itmax,K,it_init=None, \
        bounds=None,DirDoF=None):
    print "Gradient Descent Simulated Annealing..."
    N = len(x_init)
    min_x = np.copy(x_init)
    min_Func= Func(x_init)
    Energy = min_Func

    if it_init==None:
        it_init=0

    min_it=it_init

    DoF = range(N)
    if DirDoF!=None:
        for i in DirDoF:
            DoF.remove(i)

    print "Initial Gradient = ", np.linalg.norm(Jac(x_init)), "Initial Energy = ", Energy
    for j in range(K):
        it = it_init
        res = Func(x_init)
        x = np.copy(x_init)
        while ( (it < itmax) ):
            dt = 1.0/np.log(float(it+2))**0.5
            sigma =amp*np.sqrt(gamma/(np.log(it+2)*dt))*1.0/float(it+2)
            b = Jac(x)
            new_x = x - dt/gamma*b + sigma*np.sqrt(dt)/gamma*rd.normal(0.0,1.0,N)

            if (bounds!=None and len(bounds)==2 ):
                for i in DoF:
                    if (new_x[i]>bounds[0] and new_x[i]<bounds[1]):
                        x[i]=new_x[i]
            else:
                x = np.copy(new_x)

            it+=1
            PrevEnergy = Energy
            Energy = Func(x)
            if abs(PrevEnergy-Energy)/abs(Energy) < tol:
                break


        res = np.linalg.norm(Jac(x))
        Energy = Func(x)
        print "\t At iteration ",it,", Gradient = ",res, "Energy = ", Energy
        if Energy < min_Func:
            min_Func = Energy
            min_x = np.copy(x)
            min_it=it


    return [min_x,min_Func,min_it]
