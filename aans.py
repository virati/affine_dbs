#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:25:54 2019

@author: virati
AANS Submission
"""

from lie_lib import *
from dyn_lib import *
import numpy as np

# First, we set up our dynamics matrix
f = f1
h = h1

corr_matrix = jcb(f)

readout = L_d(f,h)

x = np.array([1.,2.,3.])

print('f')

for ii in np.linspace(-1,1,10):
    for jj in np.linspace(-1,1,10):
        for kk in np.linspace(-1,1,10):
            readout = L_d(f,h)
            
            xpt = np.array((ii,jj,kk))
            
            print('f at x')
            print((f(xpt)))
            print('Jacobian of f')
            print(corr_matrix(xpt))
            print('h at x')
            print(h(xpt))
            print('h along f')
            print(readout(xpt))