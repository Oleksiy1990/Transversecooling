# This code describes transverse cooling in 1D 

# x is the longitudinal direction, it's there to just let the atoms 
# move across the laser beam. The transverse cooling beam shines in y-dir




import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
from scipy.integrate import odeint
from os.path import isfile
import sys
import tables
import itertools

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
    
    rad_x : x-position from the center ("radius")
    rad_y : y-position from the center ("radius")
    wx0 : smallest beam waist in x ("radius", so 1/2 of the full waist)
    wy0 : smallest beam waist in y ("radius", so 1/2 of the full waist)
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
detun_fractionGamma = 0.5
detun = -blueGamma*detun_fractionGamma #beam detuning
lam = 460.7e-9 #460.7 nm
timeStop=0.2*10**-3
timePts=10**5
time=np.linspace(0.,timeStop,timePts)

z_init = np.array([-50e-3]) # in all initial conditions in this block 
speed_init = np.array([550])


 
#angle_init_mrad = 8
#angle_init_deg = 5
#angle_init=angle_init_deg*(np.pi/180) #conversion to radians 
#vx_init = speed_init*np.cos(angle_init)
#vy_init = speed_init*np.sin(angle_init)


"""
 Lists of parameters which we vary in simulations 
 
"""
alpha_angle = np.array([0.5,1,1.5,2,2.5,3])*(np.pi/180.) #degrees 
beta_angle = np.array([0,15,30,45])*(np.pi/180.) #degrees 

a_width_mm = np.array([2,5,8,11,14,17]) # [mm] long axis of the ellipse radius
b_width_mm = np.array([1,2,3,4,5]) # [mm] short axis of the ellipse radius 
a_width = np.array([2,5,8,11,14,17])*1e-3
b_width = np.array([1,2,3,4,5])*1e-3

power_laser_mW = np.array([15,20,25,30]) # [mW]
power_laser = power_laser_mW*1e-3

rad_position = np.array([1,2,3,4]) # [mm]
position_angle = np.array([0,15,30,45])*(np.pi/180.)

#initial positions of atoms
x_init = np.insert([rad_position*np.cos(a)*1e-3 for a in position_angle],0,0)
y_init = np.insert([rad_position*np.sin(a)*1e-3 for a in position_angle],0,0)

#initial velocities of atoms
vz_init = speed_init*np.cos(alpha_angle)
vx_init = np.insert([speed_init*np.sin(alpha)*np.cos(beta) for alpha in alpha_angle for beta in beta_angle],0,0)
vy_init = np.insert([speed_init*np.sin(alpha)*np.sin(beta) for alpha in alpha_angle for beta in beta_angle],0,0)

initialconditions = np.array(list(itertools.product(x_init,vx_init,y_init,vy_init,z_init,vz_init))) #this should be correct

#print(initialconditions.shape)
#
#print(x_init)
#print(y_init)


#sys.exit(0)

# Organizing the PyTables HDF5 file to save the data

class Simulation_output(tables.IsDescription):
    final_pos_xy = tables.Float64Col()
    final_vx = tables.Float64Col()   
    final_vy = tables.Float64Col()
    final_speed_xy = tables.Float64Col()
    init_x = tables.Float64Col()
    init_y = tables.Float64Col()   
    init_vx = tables.Float64Col()
    init_vy = tables.Float64Col()
    

class Descr(tables.IsDescription):
    detuning = tables.Float64Col()
    a_width = tables.Float64Col()
    b_width = tables.Float64Col()
    power = tables.Float64Col()
    init_z = tables.Float64Col()
    speed_init = tables.Float64Col()
    
class Timecheck(tables.IsDescription):
    time = tables.Float64Col()    
    x = tables.Float64Col()
    vx = tables.Float64Col()
    y = tables.Float64Col()
    vy = tables.Float64Col()
    z = tables.Float64Col()
    vz = tables.Float64Col()
    


#These are the parameters that are chosen for a simulation so that I don't just 
#put in values inside the code, which is unclear
power_index = 0 #we run the simulation for this entry in the power_laser vector
a_width_index = 0
b_width_index = 0

for a_width_index in range(len(a_width)):
    for b_width_index in range(len(b_width)): 
        

        file_save = tables.open_file("resultsTC/pow%.imWspeed%.idet1.hdf5"%(power_laser_mW[power_index],speed_init),mode="a",title= "Transverse cooling simulation, detuning = %.3f Gamma"%detun_fractionGamma)
        
        grp_sim = file_save.create_group("/","a%.ib%.i"%(a_width_mm[a_width_index],b_width_mm[b_width_index]))
        grp_descr = file_save.create_group("/","a%.ib%.idescr"%(a_width_mm[a_width_index],b_width_mm[b_width_index]),title="Description of the simulation with the correspondidng title")
        grp_timecheck = file_save.create_group("/","a%.ib%.itimecheck"%(a_width_mm[a_width_index],b_width_mm[b_width_index]),title="Saving one full example solution for every timestep")
        
        
        tbl_descr = file_save.create_table(grp_descr,"A",Descr,"Description of the simulation")
        tbl_descr.attrs.units = "All units in the Description table are SI"
        descr_data = tbl_descr.row
        
        descr_data["detuning"] = detun
        descr_data["a_width"] = a_width[a_width_index]
        descr_data["b_width"] = b_width[b_width_index]
        descr_data["power"] = power_laser[power_index]
        descr_data["init_z"] = z_init[0]
        descr_data["speed_init"] = speed_init[0]
        descr_data.append()
        
        tbl_descr.flush()
        
        
        
        
        print("Solving for a = %.i out of %.i and b = %.i out of %.i"%(a_width_index,len(a_width),b_width_index,len(b_width)))
        #print("Done %.i out of %.i total"%(a_width_index+b_width_index,len(a_width)+len(b_width)))
        tbl_results = file_save.create_table(grp_sim,"A",Simulation_output,"Results for the given beam width")
        output = tbl_results.row
        
        
        for num,inits in enumerate(initialconditions[0:50]):
            
        
            params = [blueKvec,blueGamma,detun,a_width[a_width_index],b_width[b_width_index],power_laser[power_index],lam]
            #print("Solving for initial conditions %.i out of %.i"%(counter,len(initialconds_red)))
            psoln = odeint(diffeqs,inits,time,args=(params,))
            
            #print("Saving data for initial conditions %.i out of %.i"%(counter,len(initialconds_red)))
        
           
            output["final_pos_xy"] = np.sqrt(psoln[-1,0]**2+psoln[-1,2]**2)
            output["final_vx"] = psoln[-1,0]   
            output["final_vy"] = psoln[-1,2] 
            output["final_speed_xy"] = np.sqrt(psoln[-1,1]**2+psoln[-1,3]**2) 
            output["init_x"] = inits[0] 
            output["init_y"] = inits[2]
            output["init_vx"] = inits[1] 
            output["init_vy"] = inits[3]
            output.append()
        
        tbl_results.flush()
        
        
        timecheck_table = np.column_stack((time,psoln[:,0],psoln[:,1],psoln[:,2],psoln[:,3],psoln[:,4],psoln[:,5]))
        arr_timecheck = file_save.create_array(grp_timecheck,"A",obj=timecheck_table,title="Example solution, all timesteps saved")
        arr_timecheck.attrs.columnorder = "(time[s],x[m],vx[m/s],y[m],vy[m/s],z[m],vz[m/s])"
        arr_timecheck.flush()
        
        
        
        
        del tbl_results
        
        # This is now a correct way to save the results into HDF5, but I need to check the equations
            
            
           
        
        file_save.close()
sys.exit(0)


#---------------------------------------------------------

coords_and_vel = tbl.row

print("Getting ready to solve")
solution_red = odeint(diffeqs_red, initialconds_red[0], t, args=(parameters_red,),mxstep=10**9)

print("Solved! Getting ready to write data")

for i in range(len(solution_red)):
    coords_and_vel["x_pos"] = solution_red[i,0]   
    coords_and_vel["vx"] = solution_red[i,1] 
    coords_and_vel["y_pos"] = solution_red[i,2] 
    coords_and_vel["vy"] = solution_red[i,3] 
    coords_and_vel["z_pos"] = solution_red[i,4] 
    coords_and_vel["vz"] = solution_red[i,5] 
    coords_and_vel.append()

tbl.flush()






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


#test




