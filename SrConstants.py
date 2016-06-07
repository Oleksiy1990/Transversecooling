import numpy as np
from pyConstants import * 

mSr84 = 84*amu
mSr86 = 86*amu
mSr87 = 87*amu
mSr88 = 88*amu

blueInvCm = 21698.452 #NIST database
blueInvM = 100*blueInvCm
blueKvec = 2*np.pi*blueInvM



blueGamma = 2*np.pi*30.5*10e6 #Hz
blueIsat_mW = 40.7 #mW/cm^2
blueIsat = 40.7*10 #W/m^2
