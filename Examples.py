#! /usr/bin/python2.7

"""
Non convex examples
"""

#from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy.random as rd
import os
import sys
from StoDescLib import *



def f(xt):
    return 0.5*xt[0]**2.0+0.5*(1.0-np.cos(2*xt[0]))+xt[1]**2.0
def df(xt):
    return np.array([2.0*0.5*xt[0]+2.0*0.5*np.sin(2*xt[0]), 2.0*xt[1]])

K=10
itmax=1000
tol=1E-5
lda=[0.1,0.1,0.1]
x0 = np.array([3.0,-5.0])
v0 = np.copy(x0)
z0 = np.copy(x0)


gamma = 1.0
amp = 1.0
[x1,min1,it1] = GradientNoise(f,df,gamma,amp,tol,x0,itmax,K)
[x2,v2,min2,it2] = Moment1(f,df,lda[0],gamma,amp,tol,x0,v0,itmax,K)
[x3,v3,z3,min3,it3] = Moment2(f,df,lda,gamma,amp,tol,x0,v0,z0,itmax,K)


print "Gradient with Noise : ", "Solution = ", x1, "Minimum = ", min1, " in ", it1, " iterations"
print "Moment 1 : ", "Solution = ", x2, "Minimum = ", min2, " in ", it2, " iterations"
print "Moment 2 : ","Solution = ", x3, "Minimum = ", min3, " in ", it3, " iterations"
