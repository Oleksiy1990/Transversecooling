import numpy as np 
import matplotlib.pyplot as plt 

import os
from os.path import isfile

simulations_dir = ""
powers = range(100)

powers_calc = [x for x in powers if isfile(simulations_dir+"power"+str(x)+".txt")]
powers_calc.sort()

files = [simulations_dir+"power"+str(x)+".txt" for x in powers_calc]
#print(files)

data = np.array([np.loadtxt(file) for file in files])


fig1 = plt.figure(figsize=(15,8))
ax1 = fig1.add_subplot(211)
for i in range(len(data)):
	ax1.plot(2*10**3*data[i,:,0], 10**3*data[i,:,1],"o") #convert everything to mm
	# beam width is now given as the real width, NOT radius
	

ax1.set_xlabel('beam width [mm]')
ax1.set_ylabel('y-final [mm]')
ax1.legend(["power = "+str(x)+" mW" for x in powers_calc],loc="lower right")

ax2 = fig1.add_subplot(212)

for i in range(len(data)):
	ax2.plot(2*10**3*data[i,:,0], data[i,:,2],"o") #convert beam rad to mm
	# beam width is now given as the real width, NOT radius

ax2.set_xlabel('beam width [mm]')
ax2.set_ylabel('vy [m/s]')
ax2.legend(["power = "+str(x)+" mW" for x in powers_calc],loc="lower right")

fig1_description = """
Simulation results for different beam widths
"""

fig1.text(0,0,fig1_description)
#fig.tight_layout()
fig1.savefig("resultsplot.png")
fig1.savefig("resultsplot.eps")

plt.show()