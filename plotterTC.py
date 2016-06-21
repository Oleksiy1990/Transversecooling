import tables
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import sys
#import matplotlib.animation as animation
import itertools

filename = "C:/Users/Oleksiy/Desktop/Code/Transversecooling/resultsTC/pow30mWspeed550det1.hdf5"
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
    df1 = df1[df1['init_x'] < 1e-3]
    df1 = df1[df1['init_y'] < 1e-3]
    df1 = df1[df1['init_vy'] < 1e-3]
    #df1 = df1[df1['init_vx'] > 2]
    #df1 = df1[df1['init_vx'] < 10]
    df1 = df1[df1['final_speed_xy'] < 1.5]
    #print(df1)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.scatter(np.sqrt(df1["init_vx"]**2 + df1["init_vx"]**2),df1["final_speed_xy"])
    ax.set_title("a%.i b%.i"%ab_combinations[index])
    plt.show()
sys.exit(0)


"""


Notes: 
"index" apparently means rows and "columns" are as it sounds in Pandas

"""

df1 = df1[df1['final_speed_xy'] < 2]
q = df1[df1['final_speed_xy'] > 1]
#q = df1.loc[lambda df1: df1.final_speed_xy < 9, :]
print(q)

sys.exit(0)


"""

Loading data from any given node
"""


nodes = ["/a%.ib%.i/A"%(u[0],u[1]) for u in itertools.product(a_widths_mm,b_widths_mm)]
print(nodes)


datalist = [load_node(node) for node in nodes] #sets of 5

print(datalist[1])






condition1 = [data[0]<1e-3 for data in datalist]
condition2 = [data[1]<1e-3 for data in datalist]
condition3 = [data[3] <1e-3 for data in datalist]
condition4 = [(7<data[2]) & (data[2]<9) for data in datalist]
condition_works = [data[4]<=1.5 for data in datalist]
condition_fails = [data[4]>=1.5 for data in datalist]

print(len(condition1))
u = list(itertools.product(a_widths_mm,b_widths_mm))


data_works = [np.extract((condition1[num] and condition2[num] and condition3[num] and condition4[num] and condition_works[num]),data[4]) for num,data in enumerate(datalist)]
fig = plt.figure()
ax1 = fig.add_subplot(211)



ax1.scatter([a[0] for a in u],[a[1] for a in u],c="b",label='first')
#ax2 = fig.add_subplot(212)
#ax2.scatter(np.extract(condition_fails,data[2]),np.extract(condition_fails,data[3]),c="r",label='second')
plt.legend(loc='upper left');
plt.show()

#condition = np.sqrt(vx_in**2+vy_in**2)<10
#x = np.extract(condition,vx_in)
#y = np.extract(condition,vy_in)
#u = np.extract(condition,speed_fin)
#plt.scatter(np.sqrt(x**2+y**2),u)
#plt.scatter(np.sqrt(vx_in**2+vy_in**2),speed_fin)
plt.show()

sys.exit(0)




for count,node in enumerate(datafile.iter_nodes(data_group,classname="Table")):
	xp_init.append(datafile.get_node(node).read(field="x_pos")[0])
	xp_final.append(datafile.get_node(node).read(field="x_pos")[-1])
	print("The count x_pos is %.i"%count)
	# if count == 50:
	# 	break

for count,node in enumerate(datafile.iter_nodes(data_group,classname="Table")):
	yp_init.append(datafile.get_node(node).read(field="y_pos")[0])
	yp_final.append(datafile.get_node(node).read(field="y_pos")[-1])
	print("The count y_pos is %.i"%count)
	# if count == 50:
	# 	break

for count,node in enumerate(datafile.iter_nodes(data_group,classname="Table")):
	zp_init.append(datafile.get_node(node).read(field="z_pos")[0])
	zp_final.append(datafile.get_node(node).read(field="z_pos")[-1])
	print("The count z_pos is %.i"%count)
	# if count == 50:
	# 	break

for count,node in enumerate(datafile.iter_nodes(data_group,classname="Table")):
	xv_init.append(datafile.get_node(node).read(field="vx")[0])
	xv_final.append(datafile.get_node(node).read(field="vx")[-1])
	print("The count vx is %.i"%count)
	# if count == 50:
	# 	break

for count,node in enumerate(datafile.iter_nodes(data_group,classname="Table")):
	yv_init.append(datafile.get_node(node).read(field="vy")[0])
	yv_final.append(datafile.get_node(node).read(field="vy")[-1])
	print("The count vy is %.i"%count)
	# if count == 50:
	# 	break

for count,node in enumerate(datafile.iter_nodes(data_group,classname="Table")):
	zv_init.append(datafile.get_node(node).read(field="vz")[0])
	zv_final.append(datafile.get_node(node).read(field="vz")[-1])
	print("The count vz is %.i"%count)
	# if count == 50:
	# 	break

print("calculating radii")

np.savetxt("xp_init.txt",xp_init)
np.savetxt("xp_final.txt",xp_final)

np.savetxt("yp_init.txt",yp_init)
np.savetxt("yp_final.txt",yp_final)

np.savetxt("zp_init.txt",zp_init)
np.savetxt("zp_final.txt",zp_final)

np.savetxt("xv_init.txt",xv_init)
np.savetxt("xv_final.txt",xv_final)

np.savetxt("yv_init.txt",yv_init)
np.savetxt("yv_final.txt",yv_final)

np.savetxt("zv_init.txt",zv_init)
np.savetxt("zv_final.txt",zv_final)


init_radii = np.sqrt(np.power(xp_init,2)+np.power(yp_init,2)+np.power(zp_init,2))
final_radii_woZ = np.sqrt(np.power(xp_final,2)+np.power(yp_final,2))

init_velocities = np.sqrt(np.power(xv_init,2)+np.power(yv_init,2)+np.power(zv_init,2))


print("calculating captured and lost")
captured_p = [init_radii[i] for i in range(len(init_radii)) if final_radii_woZ[i] <= (10**-4)]
captured_v = [init_velocities[i] for i in range(len(init_velocities)) if final_radii_woZ[i] <= (10**-4)]
lost_p = [init_radii[i] for i in range(len(init_radii)) if final_radii_woZ[i] >= (10**-4)]
lost_v = [init_velocities[i] for i in range(len(init_velocities)) if final_radii_woZ[i] >= (10**-4)]

print("plotting")
fig = plt.figure(1, figsize=(8,8))

# Plot theta as a function of time
ax1 = fig.add_subplot(211)
ax1.scatter(np.multiply(captured_p,1000), captured_v)
ax1.set_title("Captured atoms")
ax1.set_xlabel('Initial radius [mm]')
ax1.set_ylabel("Initial speed [m/s]")

# Plot omega as a function of time
ax2 = fig.add_subplot(212)
ax2.scatter(np.multiply(lost_p,1000), lost_v)
ax2.set_title("Lost atoms")
ax2.set_xlabel('Initial radius [mm]')
ax2.set_ylabel("Initial speed [m/s]")

print("saving jpg")
fig.savefig("testimg_Zignored.jpg")

print("saving eps")
fig.savefig("testimg_Zingored.eps")
plt.tight_layout()
plt.show()

sys.exit(0)

#time = np.array([datafile.get_node(node).read(field="time") for node in datafile.iter_nodes(data_group,classname="Table")])
xp = np.array([np.array(datafile.get_node(node).read(field="x_pos"))[0] for node in datafile.iter_nodes(data_group,classname="Table")])
yp = np.array([np.array(datafile.get_node(node).read(field="y_pos"))[0] for node in datafile.iter_nodes(data_group,classname="Table")])
zp = np.array([np.array(datafile.get_node(node).read(field="z_pos"))[0] for node in datafile.iter_nodes(data_group,classname="Table")])
xv = np.array([np.array(datafile.get_node(node).read(field="vx"))[0] for node in datafile.iter_nodes(data_group,classname="Table")])
yv = np.array([np.array(datafile.get_node(node).read(field="vy"))[0] for node in datafile.iter_nodes(data_group,classname="Table")])
zv = np.array([np.array(datafile.get_node(node).read(field="vz"))[0] for node in datafile.iter_nodes(data_group,classname="Table")])
datafile.close()

sys.exit(0)
# print(yp.shape)
# start_x=xp[:,0]
# start_y=yp[:,0]
# start_z=zp[:,0]
# print(start_x,start_y,start_z)

dist_start = np.sqrt(xp[:,0]**2 + yp[:,0]**2 + zp[:,0]**2)
speed_start = np.sqrt(xv[:,0]**2 + yv[:,0]**2 + zv[:,0]**2)

dist_end = np.sqrt(xp[:,-1]**2 + yp[:,-1]**2 + zp[:,-1]**2)
speed_end = np.sqrt(xv[:,-1]**2 + yv[:,-1]**2 + zv[:,-1]**2)


captured_p = [dist_start[i] for i in range(len(dist_start)) if dist_end[i] <= (10**-4)]
captured_v = [speed_start[i] for i in range(len(speed_start)) if dist_end[i] <= (10**-4)]
lost_p = [dist_start[i] for i in range(len(dist_start)) if dist_end[i] >= (10**-4)]
lost_v = [speed_start[i] for i in range(len(speed_start)) if dist_end[i] >= (10**-4)]

fig = plt.figure(1, figsize=(8,8))

# Plot theta as a function of time
ax1 = fig.add_subplot(211)
ax1.scatter(np.multiply(captured_p,1000), captured_v)
ax1.set_title("Captured atoms")
ax1.set_xlabel('Initial radius [mm]')
ax1.set_ylabel("Initial speed [m/s]")

# Plot omega as a function of time
ax2 = fig.add_subplot(212)
ax2.scatter(np.multiply(lost_p,1000), lost_v)
ax2.set_title("Lost atoms")
ax2.set_xlabel('Initial radius [mm]')
ax2.set_ylabel("Initial speed [m/s]")


plt.tight_layout()
plt.show()

print(dist_end)

# fig, ax = plt.subplots()
# scat = ax.scatter(dist_start,speed_start)

# plt.show()


sys.exit(0)
#fullarray = np.hstack((coords,results)) # align in columns the original coordinate and data arrays

captured = [item for item in fullarray if (item[6]**2+item[8]**2+item[10]**2) <= (10**-4)**2]
lost = [item for item in fullarray if (item[6]**2+item[8]**2+item[10]**2) >= (10**-4)**2]

init_rad_capt = [np.sqrt(item[0]**2 + item[2]**2 + item[4]**2) for item in captured]
init_speed_capt = [np.sqrt(item[1]**2 + item[3]**2 + item[5]**2) for item in captured]

init_rad_lost = [np.sqrt(item[0]**2 + item[2]**2 + item[4]**2) for item in lost]
init_speed_lost = [np.sqrt(item[1]**2 + item[3]**2 + item[5]**2) for item in lost]

# Plot results
fig = plt.figure(1, figsize=(8,8))

# Plot theta as a function of time
ax1 = fig.add_subplot(211)
ax1.scatter(np.multiply(init_rad_capt,1000), init_speed_capt)
ax1.set_title("Captured atoms")
ax1.set_xlabel('Initial radius [mm]')
ax1.set_ylabel("Initial speed [m/s]")

# Plot omega as a function of time
ax2 = fig.add_subplot(212)
ax2.scatter(np.multiply(init_rad_lost,1000), init_speed_lost)
ax2.set_title("Lost atoms")
ax2.set_xlabel('Initial radius [mm]')
ax2.set_ylabel("Initial speed [m/s]")


plt.tight_layout()
plt.show()