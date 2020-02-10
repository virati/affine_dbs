#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 08:40:29 2019

@author: virati
Vineet Tiruvadi
virati@gmail.com

This is a script for LIE ANALYSIS of drift and control dynamics
Run scripts with specific functions for dynamics/control and plotting
"""

#%%
# Our functions of interest HERE
import sys
sys.path.append('../src')
sys.path.append('/home/virati/Dropbox/projects/Research/DBSControl/autoLie/src')

from lie_lib import *
from lie_plots import *
import networkx as nx
import pdb
import mayavi
import lie_plots as lip

#if __name__ == '__main__':
@operable
def f1(x):
    #return np.array([-x[1],-x[0],-x[2] + x[1]])
    return np.array([-x[1]**2 + x[2],-x[0]**3,-x[2]**2 + x[1]])

@operable
def f2(x):
    return np.array([-np.sin(x[1]), -5*x[0]**2, -np.sin(x[2] - x[1])])

@operable
def f3(x):
    return -np.array([(x[0])*(x[0]-2)*(x[0]+2),x[1]**2,x[2]**2])

@operable
def f4(x):
    return -np.array([x[0],x[1],x[2]])
@operable
def f5(x):
    return -np.array([np.sin(x[2]),np.sin(x[0]),np.sin(x[1])])
    #return -np.array([x[0] + x[1], x[1]*x[2],x[2]**2])
@operable
def f6(x):
    return -np.array([x[0],x[1],x[2]])

@operable
def f7(x,args):
    return -np.array([x[0]**3 - 4*x[0]**2,x[2]**3 - 2*x[1]**2,x[1]**3 - 3*x[2]])

@operable
def f8(x,args):
    return -np.array([x[1] * x[2],x[2] * (1.0 + x[1] - x[2]),x[1] * (-1.0 + x[2] + x[1])])

class f_net:
    def __init__(self,D):
        self.D = D
        
@operable        
def net_dyn(x,D):
    new_x = np.swapaxes(np.array(x),0,2)
    D = np.array(D)
    a1 = np.dot(D.T,new_x)
    a2 = np.swapaxes(np.sin(a1),0,2)
    a3 = np.dot(D,a2)
    
    return a3

@operable
def g(x):
    #return np.sin(x + np.pi/2)
    #return np.array([np.sin(x[1]/10),np.cos(x[0]*x[1]),np.sin(x[2])])
    return np.array([x[1],0*x[1],0*x[1]])

@operable
def h(x):
    return 2*x[0] + 3*x[2]


def network_dynamics_example():
    use_func = f7
    x_ = np.linspace(-10,10,20)
    y_ = np.linspace(-10,10,20)
    z_ = np.linspace(-10,10,20)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    coords = (x,y,z)
    
    dyn_field = use_func([x,y,z])
    plot_field(dyn_field,coords,normfield=False)
    
    args = {'x':x,'y':y,'z':z}
    zeros = f_points(use_func,args,epsilon=0.5)
    fps = np.zeros(zeros.shape)
    fps[zeros == True] = 1

    fps_display = points3d(x[zeros == True],y[zeros == True],z[zeros == True],colormap='seismic',scale_factor=0.2)

def simple_control_example():
    drift = f7
    control = f8
    
    #print('Dynamics: ' + )
    mayavi.mlab.figure(bgcolor=(0.0,0.0,0.0))
    
    x_ = np.linspace(-10,10,10)
    y_ = np.linspace(-10,10,10)
    z_ = np.linspace(-10,10,10)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    coords = (x,y,z)
    
    dyn_field = drift([x,y,z],[0])
    lip.plot_field(dyn_field,coords,normfield=False,color=(1.0,0.0,0.0))
    
    ctrl_field = control([x,y,z],[0])
    lip.plot_field(ctrl_field,coords,normfield=False,color=(0.0,0.0,1.0))
    
    #We do a Lie dot product
    y_dot = L_dot(control,drift,order=1)
    plot_LD(y_dot,normalize=True)
    

def simple_dynamics_example():
    use_func = f8
    x_ = np.linspace(-10,10,20)
    y_ = np.linspace(-10,10,20)
    z_ = np.linspace(-10,10,20)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    coords = (x,y,z)
    
    dyn_field = use_func([x,y,z])
    plot_field(dyn_field,coords,normfield=False)
    
    args = {'x':x,'y':y,'z':z}
    zeros = f_points(use_func,args,epsilon=2)
    fps = np.zeros(zeros.shape)
    fps[zeros == True] = 1

    fps_display = points3d(x[zeros == True],y[zeros == True],z[zeros == True],colormap='seismic',scale_factor=0.2)

def simple_trajectory():
    x0 = np.array([0.0,0.0,0.0])
    usefunc = f8
    
    x_next = [np.copy(x0)]
    for ii in range(1000):
        x_next.append(1/1000000*usefunc(x_next[-1]))
        
    x_next = np.array(x_next)
    test = plot3d(x_next[:,0],x_next[:,1],x_next[:,2],tube_radius=0.02)
    
    #plt.plot(x_next[:,2])
    
def vector_example():
    y_dot = L_d(g,f,order=1)
    
    x_ = np.linspace(-10,10,20)
    y_ = np.linspace(-10,10,20)
    z_ = np.linspace(-10,10,20)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    coords = (x,y,z)
    
    dyn_field = f([x,y,z])
    ctrl_field = g([x,y,z])
    
    plot_fields(dyn_field,ctrl_field,coords,normfield=True)
    #plot_Ldot(y_dot)

def scalar_example():
    x_0 = np.array([1.,2.,5.])
    y_dot = L_d(h,f,order=1)
    print(np.sum(y_dot(x_0)))
    
    x_ = np.linspace(-10,10,20)
    y_ = np.linspace(-10,10,20)
    z_ = np.linspace(-10,10,20)
    
    x,y,z = np.meshgrid(x_,y_,z_,indexing='ij')
    
    
    dyn_field = f([x,y,z])
    read_field = h([x,y,z])
    #This plots the dynamics field first
    obj = quiver3d(x,y,z,dyn_field[0,:,:,:],dyn_field[1,:,:,:],dyn_field[2,:,:,:])
    obj2 = points3d(x,y,z,read_field[:,:,:],colormap='seismic',scale_factor=0.01)
    #plot_LD(y_dot)


#%%
    
if __name__ == '__main__':
    ## EXAMPLE 1
    #simple_control_example()
    #simple_trajectory()
    
    # EXAMPLE 2
    simple_control_example()
    mlab.show()