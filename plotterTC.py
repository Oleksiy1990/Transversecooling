import tables
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import sys
#import matplotlib.animation as animation
import itertools

filename = "C:/Users/Oleksiy/Desktop/Code/Transversecooling/resultsTC/pow30mWspeed600det1.hdf5"
#filename="/Users/oleksiy/Desktop/PythonCode/Transversecooling/resultsTC/pow30mWspeed550det1.hdf5"
datafile = tables.open_file(filename,"r")


a_widths_mm = [2,5,8,11,14,17]
b_widths_mm = [1,2,3,4,5]

ab_combinations = list(itertools.product(a_widths_mm,b_widths_mm))

node = "/a14b5/A"
nodes = ["/a%.ib%.i/A"%(u[0],u[1]) for u in ab_combinations]

#def load_node(node):
#    x_in = datafile.get_node(node).read(field="init_x")
#    y_in = datafile.get_node(node).read(field="init_y")
#    vx_in = datafile.get_node(node).read(field="init_vx")
#    vy_in = datafile.get_node(node).read(field="init_vy")
#    speed_fin = datafile.get_node(node).read(field="final_speed_xy")
#    return [x_in,y_in,vx_in,vy_in,speed_fin]



loaded_nodes = [pd.read_hdf(filename,key=node,mode="r") for node in nodes] # we have to give it a file 
#name, the ndoe to read, and tell it to do it read-only



"""
We load the HDF5 which has beed read before as a DataFrame (using Pandas)

We then check condition by condition to narrow down and select the data that we like 
"""
for index,u in enumerate(loaded_nodes):

    df1 = pd.DataFrame(u,columns=list(("init_x","init_y","init_vx","init_vy","final_speed_xy"))) 
    #df1 = df1[df1['init_x'] < 1e-3]
    #df1 = df1[df1['init_y'] < 1e-3]
    #df1 = df1[df1['init_vy'] < 1e-3]
    #df1 = df1[df1['init_vx'] > 2]
    #df1 = df1[df1['init_vx'] < 10]
    df1 = df1[df1['final_speed_xy'] < 1.5]
    #print(df1)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(np.sqrt(df1["init_vx"]**2 + df1["init_vx"]**2),df1["final_speed_xy"])
    ax.set_title("a%.i b%.i"%ab_combinations[index])
    ax.set_xlabel("Init speed [m/s]")
    ax.set_ylabel("Final speed [m/s]")
    plt.show()


"""


Notes: 
"index" apparently means rows and "columns" are as it sounds in Pandas

"""

