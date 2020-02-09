#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:35:02 2019

@author: virati
The main file for the Lie-algebra, neural network project with Jirsa Lab
This is the big kahoona
"""

import sys
sys.path.append('../src/')

from lie_lib import *
import networkx as nx
import ipdb
import autograd.numpy as np
from mayavi import mlab

''' First, we'll define the functions we're interested in '''
def integrator(f,state,params):
    dt=0.01
    k1 = f(state,params) * dt
    k2 = f(state + .5*k1,params)*dt
    k3 = f(state + .5*k2,params)*dt
    k4 = f(state + k3,params)*dt
    
    new_state = state + (k1 + 2*k2 + 2*k3 + k4)/6
    #new_state += np.random.normal(0,10,new_state.shape) * dt
    
    return new_state

class dyn_system:
    f_drift = []
    g_ctrl = []
    u = []
    
    def __init__(self):
        pass
        
    def simulate(self,state,tsteps=np.linspace(0,10,1000)):
        #f = self.f_drift + self.g_ctrl * self.u
        raster = []
        for ii,time in enumerate(tsteps):
            x_new = integrator(self.f_drift, state, self.P) + integrator(self.g_ctrl,state,self.P)
            raster.append(x_new)
            
        self.sim_raster = raster

class control_system(dyn_system):
    def __init__(self):
        #set our drift dynamics
        self.f_drift = f_trivial
        self.g_ctrl = g_mono
        self.u = u_step
        self.h = h_single
        
        #set our graph
        n_elements = 10
        n_regions = int(np.floor(n_elements/2))
        self.G = nx.random_regular_graph(4, n_elements)
        self.L = nx.linalg.laplacian_matrix(self.G).todense()
        self.D = np.array(nx.linalg.incidence_matrix(self.G).todense())
        self.D = np.diag(np.ones(shape=(n_elements,)))
        # for each of our elements, assign them to a brain region
        self.e_to_r = np.random.randint(0,n_regions,size=n_elements)
        
        #do our disease layer
        n_symp = 2
        #self.Xi = np.random.randint(0,1,size=(n_regions,n_symp))
        self.Xi = Xi_1
        
        self.P = self.L
        
        self.x_state = np.random.uniform(size=(1000,1))
        
        self.n_regions = n_regions
        self.n_symp = n_symp
        self.n_elements = n_elements
        
    def render_graph(self):
        H = self.G
        # reorder nodes from 0,len(G)-1
        G=nx.convert_node_labels_to_integers(H)
        # 3d spring layout
        pos=nx.spring_layout(G,dim=3)
        # numpy array of x,y,z positions in sorted node order
        xyz=np.array([pos[v] for v in sorted(G)])
        # scalar colors
        scalars=np.array(G.nodes())+5
        
        mlab.figure(1, bgcolor=(0, 0, 0))
        mlab.clf()
        
        pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                            scalars,
                            scale_factor=0.1,
                            scale_mode='none',
                            colormap='Blues',
                            resolution=20)
        ctrl_indices = self.g_ctrl(np.ones(shape=(xyz.shape[0],)),[0])>0
        readout_indices = h_single_vect(np.ones(shape=(xyz.shape[0],)),[0])>0
        pts_ctrl = mlab.points3d(xyz[ctrl_indices,0], xyz[ctrl_indices,1], xyz[ctrl_indices,2],
                            scalars[ctrl_indices],
                            scale_factor=0.3,
                            scale_mode='none',
                            color=(1.0,0.0,0.0),
                            resolution=20)
        
        pts_ro = mlab.points3d(xyz[readout_indices,0], xyz[readout_indices,1], xyz[readout_indices,2],
                            scalars[readout_indices],
                            scale_factor=0.3,
                            scale_mode='none',
                            color=(0.0,0.0,1.0),
                            resolution=20)
        
        pts.mlab_source.dataset.lines = np.array(G.edges())
        tube = mlab.pipeline.tube(pts, tube_radius=0.01)
        mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8),opacity=0.1)
        
        mlab.savefig('mayavi2_spring.png')
        #mlab.show() # interactive window

    
    def disease_measure(self):
        h_grad = egrad(self.h)
        Xi_grad = egrad(self.Xi)
        
        self.interact_vector = L_dot(self.h,self.Xi)
        rand_checks = np.random.uniform(-10,10,size=(self.n_elements,self.n_elements))
        is_zero = []
        measure_sets = []
        new_measure_sets = []
        for ii in range(self.n_elements):
            measure_sets.append(self.interact_vector(rand_checks[ii,:].squeeze(),[0])[0])
            
            #print(measure_set)
        
            new_measure_sets.append(np.dot(h_grad(rand_checks[ii,:],0),self.Xi(rand_checks[ii,:],0)))
            is_zero.append(new_measure_sets[-1] == 0)
        #pdb.set_trace()
        print('Measurement-Disease interaction is zero: ' + str(np.array(is_zero).all() == True))
        self.measure_sets = new_measure_sets
        #self.measure_pts = rand_checks   
        
                           
    ''' The main result from our ability to control the disease state through g'''
    def disease_control(self):
        Xi_grad = egrad(self.Xi)
        g_grad = egrad(self.g_ctrl)
        
        #
        self.interact_vector = L_d(self.g_ctrl,self.Xi)
        #choose N random vectors in the N dim space
        rand_checks = np.random.uniform(-10,10,size=(self.n_elements,self.n_elements))
        is_zero = []
        control_sets = []
        new_control_sets = []
        for ii in range(self.n_elements):
            #control_sets.append(self.interact_vector(rand_checks[ii,:].squeeze(),[0]))
            
            new_control_sets.append(np.dot(g_grad(rand_checks[ii,:],0),self.Xi(rand_checks[ii,:],0)))
            is_zero.append(new_control_sets[-1] == 0)
            #print(control_set)
        
        print('Control-Disease interaction is zero: ' + str(np.array(is_zero).all() == True))
        #If we find even a single non-zero, we know we can some control
        #If they're all zero, still unsure whether it's truly zero everywhere or if we just 'got very lucky' withour random points == criticalpts
        
        self.control_sets = new_control_sets
        
    def full_control(self):
        Xi_grad = egrad(self.Xi)
        g_grad = egrad(self.g_ctrl)
        f_grad = egrad(self.f_drift,0) #only take gradient along first argument
        
        
        self.interact_vector = L_d(self.g_ctrl,self.Xi)
        #choose N random vectors in the N dim space
        rand_checks = np.random.uniform(-10,10,size=(self.n_elements,self.n_elements))
        is_zero = []
        full_sets = []
        for ii in range(self.n_elements):
            #test = f_grad(rand_checks[ii,:],self.D)
            full_sets.append(np.dot(f_grad(rand_checks[ii,:],self.D),self.Xi(rand_checks[ii,:],0)) + np.dot(g_grad(rand_checks[ii,:],0),self.Xi(rand_checks[ii,:],0)))
            is_zero.append(full_sets[-1] == 0)
            #print(control_set)
        
        print('Dyn+Ctrl is zero: ' + str(np.array(is_zero).all() == True))
        #If we find even a single non-zero, we know we can some control
        #If they're all zero, still unsure whether it's truly zero everywhere or if we just 'got very lucky' withour random points == criticalpts
        
        self.full_sets = full_sets
        
    ''' Below isn't necessary for ASSFN project'''
    def disease_bracket(self):
        Xi_grad = egrad(self.Xi)
        g_grad = egrad(self.g_ctrl)
        f_grad = egrad(self.f_drift,0) #only take gradient along first argument
        
        
        #self.interact_vector = L_d(self.g_ctrl,self.Xi)
        #choose N random vectors in the N dim space
        rand_checks = np.random.uniform(-10,10,size=(self.n_elements,self.n_elements))
        is_zero = []
        bracket_sets = []
        for ii in range(self.n_elements):
            #test = f_grad(rand_checks[ii,:],self.D)
            bracket_sets.append(np.dot(f_grad(rand_checks[ii,:],self.D),self.g_ctrl(rand_checks[ii,:],0)) - np.dot(g_grad(rand_checks[ii,:],0),self.f_drift(rand_checks[ii,:],self.D)))
            is_zero.append(bracket_sets[-1] == 0)
            #print(control_set)
        
        print('Bracket is zero: ' + str(np.array(is_zero).all() == True))
        #If we find even a single non-zero, we know we can some control
        #If they're all zero, still unsure whether it's truly zero everywhere or if we just 'got very lucky' withour random points == criticalpts
        
        self.bracket_sets = bracket_sets
        
    def laplac(self):
        return nx.linalg.laplacian_matrix(self.G).todense()
    
def f_k(x,D):
    #x_1 = np.dot(D.T,x)
    x_1 = np.dot(D.T,x)
    x_2 = np.sin(x_1)
    x_3 = D * x_2
    
    return x_3

def f_trivial(x,D):
    filt_x = np.zeros(shape=(x.shape[0],x.shape[0]))
    filt_x[1,1] = 1.0
    filt_x[3,3] = 1.0
    filt_x[7,9] = 1.0
    
    return np.dot(filt_x.T,x)

def Xi_1(x,P):
    #return np.array([0.0,0.0,0.0,0.0,0.,0.,0.,0.,0,1.0 * x[9]])
    return np.array([0.0,0.0,0.0,0.0,0.,0.,0.,0.,0.0,1.0]) * x
    #return np.dot(np.random.randint(0,1,size=(10,P[0])),x)

def h_single_vect(x,P):
    measure_vect = np.zeros_like(x)
    measure_vect[3] = np.sin(x[3])
    return measure_vect

def h_single(x,P):
    measure_vect = np.zeros_like(x)
    measure_vect[3] = 1.0
    #pdb.set_trace()
    oscollate = np.sin(x)
    
    
    return np.dot(measure_vect.T,oscollate)

def g_mono(x,P):
    test = np.zeros(shape=(x.shape[0],x.shape[0]))
    resid = np.zeros(shape=x.shape)
    test[7,7] = 1.0
    ret_vec = np.dot(test,x) + resid
    
    return ret_vec

#%%
# We first care about the drift dynamics
#@operable
def f_hopf(x,P):
    r = x[:,0].reshape(-1,1)
    theta = x[:,1].reshape(-1,1)
    
    #Node-based dynamics done here
    # c is a function of the network inputs into the node
    neighbors = np.dot(L,r)
    
    r_dot = np.diag(np.outer(r,r - neighbors)).reshape(-1,1)
    
    #theta_dot = self.w * 1/(1-np.tanh(p[0]-self.c))
    theta_dot = 0.02 * np.exp(r)
    
    
    return np.array([r_dot,theta_dot])

def f_consensus(x,P):
    x_dot = -np.dot(P[0],x)
    
    return x_dot

def u_step(t,P):
    return (t > 5).astype(np.float)

#%% Script running code
    
if __name__=='__main__':
    
    brain = control_system()
    #brain.simulate(state=x0)
    brain.disease_control()
    brain.disease_measure()
    brain.full_control()
