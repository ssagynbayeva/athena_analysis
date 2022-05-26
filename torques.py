import matplotlib.pyplot as plt
import numpy as np
import pylab
import matplotlib.colors as mplc
from matplotlib import ticker 
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import rc
import sys
sys.path.insert(0, '$ATHENA_DIR/vis/python')
import numpy as np
from matplotlib.colors import LogNorm
from pylab import *
import struct
import array
import os
from scipy.interpolate import griddata
import h5py

# Athena++ modules
import athena_read

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

#exec(open("disk_analysis.py").read())

fname = "disk.hst"



r = np.genfromtxt(fname, unpack=True, max_rows=1)
with open(fname, 'r') as f:
    num_cols = len(f.readline().split())
    f.seek(0)
torque = np.genfromtxt(fname, unpack=True, skip_header=1, usecols=(16))
t = np.genfromtxt(fname, unpack=True, skip_header=1, usecols=(0))
dt = np.genfromtxt(fname, unpack=True, skip_header=1, usecols=(1))

rho0 = 1.
rp = 1.
omega_p = 1/(2*np.pi*3)
q = 1.5e-5
c_s = 0.03

mp = np.genfromtxt(fname, unpack=True, skip_header=1, usecols=(2))
vp = np.genfromtxt(fname, unpack=True, skip_header=1, usecols=(5))/mp

torque_zero = rho0*4*np.pi*rp**2*0.05*rp**4*q**2*(c_s)**(-2)

plt.plot(t, torque)
plt.xlabel('Time')
plt.ylabel('Torque')
plt.title('mass=1.5e-5')
plt.show()




def grav_pot_car_btoa(xca, yca, zca, xcb, ycb, zcb, gb, rsoft2):
  rsoft=np.sqrt(rsoft2)
  pot = np.zeros(len(xca))
  for i in range(len(xca)):
    dist=np.sqrt((xca[i]-xcb)**2 + (yca[i]-ycb)**2 + (zca[i]-zcb)**2)
    dos=dist/rsoft
    if(dist>=rsoft):
      pot[i]=-gb/dist
    else:
      pot[i]=-gb/dist*(dos**4-2.*dos**3+2*dos)

  return pot


'''
dphi = 1.e-3

x0 = -1
rsoft2 = 1.44e-4

vol = 2.70005e-04

torque_athena = np.zeros((len(phi), len(theta),len(x1v)))


#mp = np.genfromtxt(fname, unpack=True, skip_header=1, usecols=(2))
for i in range(len(phi)):
  for j in range(len(theta)):
    xpcar = x1v*np.sin(j)*np.cos(i)
    ypcar = x1v*np.sin(j)*np.sin(i)
    zpcar = x1v*np.cos(j)

    xmcar = x1v*np.sin(j)*np.cos(i-dphi)
    ymcar = x1v*np.sin(j)*np.sin(i-dphi)
    zmcar = x1v*np.cos(j)




    torque_athena[i][j] = -vol*(rho[i,j,:]*(grav_pot_car_btoa(xpcar,ypcar,zpcar,x0,0,0,1.,rsoft2) \
  -grav_pot_car_btoa(xmcar,ymcar,zmcar,x0,0,0,1.,rsoft2))/2.0/dphi)

torque_athena = np.sum(np.sum(torque_athena, axis=0), axis=0)




plt.plot(x1v-x0, torque_athena)
plt.xlabel('radius')
plt.ylabel('Torque')
plt.title('mass=1.5e-5')
plt.show()
'''

theeeta = np.linspace(0,np.pi,len(x1v))
phiii = np.linspace(0,np.pi,len(x1v))

x = x1v*np.cos(theeeta)
y = x1v*np.cos(theeeta)



