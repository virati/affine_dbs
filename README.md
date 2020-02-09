# Affine DBS

## Overview
Adaptive Deep Brain Stimulation using control-affine approaches
![Overview of Framework](imgs/ctrl_aff_sysdiag.png)

### Requirements
This project uses the [AutoLie library](https://github.com/virati/autoLie)

## Introduction
Control-affine frameworks enable us to bring tools from differential geometry to bear on control theory problems [].

The control-affine framework we'll use for our study is:
![]().

The brain dynamics $\dot{x}$ are determined by an intrinsic, or *drift*, dynamics $f_\mathcal{L}$.

The control signal $\vec{u}(t)$ is then coupled into our system dynamics through $g_\tau(x)$.

This structure is what enables us to then use techniques from geometric control theory to analyse how our intrinsic dynamics $f$ interact with the control coupling $g$.

**Measuring** our system is crucial. In a closed-loop control system, our control signal $\vec{u}$ is a function of our state $x$ through our measurement of that state $\vec{y}$.



## Network-Disease Model
First, we'll build the Network-Disease Model

### Brain Network
Starting with a connectivity matrix, we build a network.

We then add the dynamics. For this demomnstration we assume a fairly simple dynamics.

### Disease Network
To add the disease to our network model, we build a linear mapping between the brain state and the behavioral state.


In this model, the symptoms are related to each other only through the brain networks.

