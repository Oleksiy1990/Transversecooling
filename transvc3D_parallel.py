# This code describes transverse cooling in 1D 

# x is the longitudinal direction, it's there to just let the atoms 
# move across the laser beam. The transverse cooling beam shines in y-dir




import numpy as np 
#import matplotlib.pyplot as plt 
#import scipy as sp 
from scipy.integrate import odeint
from os.path import isfile
import sys
import tables
import itertools
from multiprocessing import Process, Lock

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





""" 

Parameters for simulation which we don't sweep
"""
detun_fractionGamma = 0.5
detun = -blueGamma*detun_fractionGamma #beam detuning
lam = 460.7e-9 #460.7 nm
timeStop=0.2*10**-3
timePts=10**5
time=np.linspace(0.,timeStop,timePts)

z_init = np.array([-50e-3]) # in all initial conditions in this block 
speed_init = np.array([600])
 



"""
 Lists of parameters which we vary in simulations 
 
"""
alpha_angle = np.array([0,0.5,1.5,2.5])*(np.pi/180.) #degrees #try deleting a few for simulation speed
beta_angle = np.array([0,15,30,45])*(np.pi/180.) #degrees 
alpha_beta = list(itertools.product(alpha_angle,beta_angle))


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

# [[speed_init*np.sin(alpha)*np.cos(beta),speed_init*np.sin(alpha)*np.sin(beta),speed_init*np.cos(alpha)] for alpha,beta in itertools.product(alpha_angle,beta_angle)]
# vz_init = speed_init*np.cos(alpha_angle)
# vx_init = np.insert([speed_init*np.sin(alpha)*np.cos(beta) for alpha in alpha_angle for beta in beta_angle],0,0)  
# vy_init = np.insert([speed_init*np.sin(alpha)*np.sin(beta) for alpha in alpha_angle for beta in beta_angle],0,0)

initialconditions = np.array(list(itertools.product(x_init,vx_init,y_init,vy_init,z_init,vz_init))) #This is in principle incorrect because all variables are NOT independent! 

print(initialconditions.shape)
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
power_index = 2 #we run the simulation for this entry in the power_laser vector
#a_width_index = 0
#b_width_index = 0

def simulation(ab_list,l):


    for ab in ab_list: # These should be the numbers to index the a and b widths to take
        
        print("Doing ab ",ab)

        final_pos_xy = []
        final_vx = []
        final_vy = []
        final_speed_xy = []
        init_x = []
        init_y = []
        init_vx = []
        init_vy = []

        for num,inits in enumerate(initialconditions):
            
        
            params = [blueKvec,blueGamma,detun,a_width[ab[0]],b_width[ab[1]],power_laser[power_index],lam]
            
            psoln = odeint(diffeqs,inits,time,args=(params,))
            
            """
            results will be in the form: 
            [final_pos_xy,final_vx,final_vy,final_speed_xy,init_x,init_y,init_vx,init_vy]

            """
            final_pos_xy.append(np.sqrt(psoln[-1,0]**2+psoln[-1,2]**2))
            final_vx.append(psoln[-1,0])
            final_vy.append(psoln[-1,2])
            final_speed_xy.append(np.sqrt(psoln[-1,1]**2+psoln[-1,3]**2))
            init_x.append(inits[0])
            init_y.append(inits[2])
            init_vx.append(inits[1])
            init_vy.append(inits[3])

        
           
            

        

        l.acquire() # l is an instance of Lock() class, and it's defined in the Main

        file_save = tables.open_file("resultsTC/pow%.imWspeed%.idet1.hdf5"%(power_laser_mW[power_index],speed_init),mode="a",title= "Transverse cooling simulation, detuning = %.3f Gamma"%detun_fractionGamma)
        grp_sim = file_save.create_group("/","a%.ib%.i"%(a_width_mm[ab[0]],b_width_mm[ab[1]]))
        grp_descr = file_save.create_group("/","a%.ib%.idescr"%(a_width_mm[ab[0]],b_width_mm[ab[1]]),title="Description of the simulation with the correspondidng title")
        grp_timecheck = file_save.create_group("/","a%.ib%.itimecheck"%(a_width_mm[ab[0]],b_width_mm[ab[1]]),title="Saving one full example solution for every timestep")
        
        tbl_results = file_save.create_table(grp_sim,"A",Simulation_output,"Results for the given beam width")
        output = tbl_results.row

        for q in range(len(final_pos_xy)):
            output["final_pos_xy"] = final_pos_xy[q]
            output["final_vx"] = final_vx[q]   
            output["final_vy"] = final_vy[q]
            output["final_speed_xy"] = final_speed_xy[q]
            output["init_x"] = init_x[q]
            output["init_y"] = init_y[q]
            output["init_vx"] = init_vx[q]
            output["init_vy"] = init_vy[q]
            output.append()
        tbl_results.flush()

        tbl_descr = file_save.create_table(grp_descr,"A",Descr,"Description of the simulation")
        tbl_descr.attrs.units = "All units in the Description table are SI"
        descr_data = tbl_descr.row
        
        descr_data["detuning"] = detun
        descr_data["a_width"] = a_width[ab[0]]
        descr_data["b_width"] = b_width[ab[1]]
        descr_data["power"] = power_laser[power_index]
        descr_data["init_z"] = z_init[0]
        descr_data["speed_init"] = speed_init[0]
        descr_data.append()
        tbl_descr.flush()
        
        print("Getting ready to save timecheck")
        timecheck_table = np.column_stack((time,psoln[:,0],psoln[:,1],psoln[:,2],psoln[:,3],psoln[:,4],psoln[:,5]))
        arr_timecheck = file_save.create_array(grp_timecheck,"A",obj=timecheck_table,title="Example solution, all timesteps saved")
        arr_timecheck.attrs.columnorder = "(time[s],x[m],vx[m/s],y[m],vy[m/s],z[m],vz[m/s])"
        arr_timecheck.flush()
        print("Done with timecheck")
        file_save.close()
        
        l.release()
               
        
        
        print("Done with ab ",ab)
        
        # This is now a correct way to save the results into HDF5, but I need to check the equations
            
            
           
        
        


if __name__ == "__main__": 
    
    
    print(len(a_width))
    print(len(b_width))
    indices_ab = np.array(list(itertools.product(range(len(a_width)),range(len(b_width)))))  
    num_processes = 7
    indices_ab_forprocesses = np.array_split(indices_ab,num_processes)
    
    l = Lock() #We need to make only one lock! 
    # So we need only one Lock class instance, apparently that's important


    process_list = []
    for q in range(num_processes):
        pr = Process(target=simulation,args=(indices_ab_forprocesses[q],l))
        process_list.append(pr)

    [pr.start() for pr in process_list]

    [pr.join() for pr in process_list]


    

    
    

    



