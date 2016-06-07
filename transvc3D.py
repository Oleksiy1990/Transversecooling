# This code describes transverse cooling in 1D 

# x is the longitudinal direction, it's there to just let the atoms 
# move across the laser beam. The transverse cooling beam shines in y-dir




import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
from scipy.integrate import odeint
from os.path import isfile
import sys

#Constants
from pyConstants import *
from SrConstants import * 



def molasses_force(k_vector_laser,linewidth,s0,detuning,velocity):
    """
    standard function to describe the optical scattering force in molasses configuraion
    http://cua.mit.edu/8.422_S07/Notes%20on%20classical%20molasses%20and%20beam%20slowing.pdf
    """

    part_minus = 1/(1+s0+((2*detuning/linewidth-2*k_vector_laser*velocity/linewidth)**2))
    part_plus = 1/(1+s0+((2*detuning/linewidth+2*k_vector_laser*velocity/linewidth)**2))
    force = hbar * k_vector_laser * (linewidth/2) * s0 * (part_minus - part_plus)
    return force



def gauss_I_2D(rad_x,rad_y,wx0,wy0,power,lam,z):
    """
    2D normalized intensity of a Gaussian laser beam
    z is the general name for the direction of propagation, it's not necessarily the z-coord 
    in a given simulation problem 
    """
    rayleigh_x = (wx0**2 * np.pi)/lam
    rayleigh_y = (wy0**2 * np.pi)/lam
    waist_x = wx0*np.sqrt(1+(z/rayleigh_x)**2)
    waist_y = wy0*np.sqrt(1+(z/rayleigh_y)**2)
    norm_prefactor = 2*power/(np.pi*waist_x*waist_y)
    intensity = norm_prefactor*np.exp(-2*(rad_x**2)/waist_x**2)*np.exp(-2*(rad_y**2)/waist_y**2)
    return intensity

def diffeqs(variables,t,params):
    """
    these are coupled equations for motion along the machine and transverse cooling
    They are written as first order equations because the Python solver 
    only takes the 1st order diff eqs -> higher-order should be manually converted to 
    systems of 1st order diff. eqs. 

    I assume that there's no cooling in x-dir. That would come from the Zeeman 
    slower beam but it doesn't matter because it can only make things better

    Isat is given in W/m^2, and we work in SI units
    """

    x,vx,y,vy,z,vz = variables # positions and velocities; z is along the line of the machine 
    k_vector_laser,linewidth,detuning,wx0,wy0,power,lam = params

    s0x = gauss_I_2D(z,y,wx0,wy0,power,lam,x)/blueIsat # here z is the wide axis, y is the short axis, x is 
    #the propagation axis
    s0y = gauss_I_2D(z,x,wx0,wy0,power,lam,y)/blueIsat # here z is the wide axis, x is the short axis, y is 
    #the propagation axis

    derivs = [vx,(1/mSr84)*molasses_force(k_vector_laser,linewidth,s0x,detuning,vx), \
    vy, (1/mSr84)*molasses_force(k_vector_laser,linewidth,s0y,detuning,vy),\
    vz, 0]
    return derivs



# Parameters for simulation which we don't sweep

detun = -blueGamma/2 #beam detuning


x_init = -50*10**-3 #all initial conditions in this block 
y_init = 0
speed_init = 550
#angle_init_mrad = 8
angle_init_deg = 5
angle_init=angle_init_deg*(np.pi/180) #conversion to radians 
vx_init = speed_init*np.cos(angle_init)
vy_init = speed_init*np.sin(angle_init)

powerlaser = 30*10**-3


# we sweep through the values of beam radii
beam_width_init = 1*10**-3
beam_width_final = 50*10**-3
beam_width_pts = 30
beamwidths = np.linspace(beam_width_init,beam_width_final,beam_width_pts)

# these are the arrays to fill with results
y_final = []
vy_final = []

# we save the simulation results and description to files
call_simulation = "power30"
folder_to_save = "simresults"+str(angle_init_deg)+"deg/"
file_to_save = folder_to_save+call_simulation+".txt"
file_to_save_descr = folder_to_save+call_simulation+"_README.txt"
file_to_save_lastimage = folder_to_save+call_simulation+"_graph.eps"
file_to_save_lastimage_png = folder_to_save+call_simulation+"_graph.png"
# heads up here because path declarations depend on operating system

# this is a check to make sure that we don't unintentionally overwrite stuff
if isfile(file_to_save) or isfile(file_to_save_descr):
    sys.exit("Change filename because otherwise you will overwrite existing data")

# sweeping through beam widths
for q in beamwidths:
    w0 = q
    params = [blueKvec,blueGamma,detun,w0,powerlaser]
    init_conds = [x_init,vx_init,y_init,vy_init]

    tStop=0.2*10**-3
    tPts=10**5
    t=np.linspace(0.,tStop,tPts)

    psoln = odeint(diffeqs,init_conds,t,args=(params,))
    y_final.append(psoln[-1,2])
    vy_final.append(psoln[-1,3])
    # print(w0)
    # print("y")
    # print(y_final)
    # print("vy")
    # print(vy_final)

# the output is a 3-column array 
output = np.column_stack((beamwidths,y_final,vy_final))

np.savetxt(file_to_save,output)
f = open(file_to_save_descr,"a")
f.write("detuning = %.3f "%(detun/blueGamma) + "Gamma \n")
f.write("laserpower = %.3f "%powerlaser + "W \n") 
f.write("inital speed = %.3f "%speed_init +"m/s \n")
f.write("inital angle with respect to x-axis =%.3f "%angle_init +"rad \n")
f.write("Format: (beam width [m],y-final[m],vy-final[m/s]) \n")
f.close()
    

fig = plt.figure(1, figsize=(8,8))

# Plot theta as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(t, psoln[:,2])
ax1.set_xlabel('time [s]')
ax1.set_ylabel('y [m]')

# Plot omega as a function of time
ax2 = fig.add_subplot(312)
ax2.plot(t, psoln[:,3])
ax2.set_xlabel('time [s]')
ax2.set_ylabel('vy [m/s]')

ax3 = fig.add_subplot(313)
ax3.plot(t, psoln[:,0])
ax3.set_xlabel('time [s]')
ax3.set_ylabel('x [s]')

fig_description = """
To show that at the end of integration the atom is already outside of 
the transverse cooling beams
"""

fig.text(0,0,fig_description)
fig.tight_layout()
fig.savefig(file_to_save_lastimage)
fig.savefig(file_to_save_lastimage_png)
#plt.show()







