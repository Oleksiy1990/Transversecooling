# example tryout of damped harmonic motion with Python integrator

# This example solves damped harmonic motion using the odeint function of 
# scipy.integrate 

import numpy as np 
import scipy as sp 
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def oscil(var,t,params):
	x, y = var
	damping, res_freq = params
	derivs = [y,-2*damping*y-(res_freq**2)*x]
	return derivs

damp = 0.2
res_freq = 2

params = [damp,res_freq]
initc = [0,3]

tStop = 20.
tInc = 0.00005 #this is still very fast, but this is already 400000 time steps
t = np.arange(0., tStop, tInc)
psoln = odeint(oscil, initc, t, args=(params,))

# Plot results
fig = plt.figure(1, figsize=(8,8))

# Plot theta as a function of time
ax1 = fig.add_subplot(211)
ax1.plot(t, psoln[:,0])
ax1.set_xlabel('time')
ax1.set_ylabel('x')

# Plot omega as a function of time
ax2 = fig.add_subplot(212)
ax2.plot(t, psoln[:,1])
ax2.set_xlabel('time')
ax2.set_ylabel('v')


plt.tight_layout()
plt.show()

