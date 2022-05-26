
import sys
sys.path.insert(0, '$ATHENA_DIR/vis/python')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pylab import *
import struct
import array
import os
from scipy.interpolate import griddata


import h5py


# setup latex fonts
# rcParams['text.usetex']=True
#rcParams['text.latex.unicode']=True


#matplotlib.rc('font', family='serif', serif='cm10')
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black','cyan','magenta'])
cmap2 = 'PuRd'
#cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['turquoise','coral','lavender'])
#cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white','coral','darkorange','turquoise','lavender'])

# Athena++ modules
import athena_read

input_filename = 'disk.out1.00009.athdf'
leveln = None

quantities=['rho','press','vel1','vel2','vel3']
with h5py.File(input_filename, 'r') as f:
      attributes = f.attrs.items()
      attrs = dict(attributes)
      level = f.attrs['MaxLevel']
      time = f.attrs['Time']

subsample = False
if leveln is not None:
    if level > leveln:
        subsample = True
    level = leveln
data = athena_read.athdf(input_filename, quantities=quantities,
    level=level, subsample=subsample)

rho=data['rho']
velr=data['vel1']
velphi=data['vel3']
veltheta=data['vel2']
press=data['press']
temp=press/rho

r=data['x1f'][:-1]
theta=data['x2f'][:-1]
phi=data['x3f'][:-1]

nx1 = attrs['RootGridSize'][0] * 2**level
nx2 = attrs['RootGridSize'][1] * 2**level
nx3 = attrs['RootGridSize'][2] * 2**level


#plot rho, v and Er, B
# make plots in r-theta plane and mid-plane

x1f=data['x1f']
x2f=data['x2f']
x3f=data['x3f']


x1v=np.zeros(nx1)
x2v=np.zeros(nx2)
x3v=np.zeros(nx3)

for i in range(nx1):
    x1v[i]=0.75*(x1f[i+1]**4.0 - x1f[i]**4.0)/(x1f[i+1]**3.0-x1f[i]**3.0)

for j in range(nx2):
    x2v[j]=((np.sin(x2f[j+1]) - x2f[j+1] * np.cos(x2f[j+1])) \
        -(np.sin(x2f[j]) - x2f[j] * np.cos(x2f[j]))) \
            / (np.cos(x2f[j]) - np.cos(x2f[j+1]))

for k in range(nx3):
    x3v[k] = 0.5 * (x3f[k+1]+x3f[k])


# now the mid-plane plots

# find the position corresponding to phiplot position
rmax = max(x1f)
thetapos=np.abs(x2v-0.5*np.pi).argmin()
rmaxpos=np.abs(x1v-rmax).argmin()


rhoslice=rho[:,thetapos,:rmaxpos]
velr_slice=velr[:,thetapos,:rmaxpos]
velphi_slice=velphi[:,thetapos,:rmaxpos]
veltheta_slice=veltheta[:,thetapos,:rmaxpos]
press_slice=press[:,thetapos,:rmaxpos]
tempslice=temp[:,thetapos,:rmaxpos]

x1f=x1f[:rmaxpos+1]

#logr=np.log10(x1f)
#x1f_norm = (x1f - np.min(x1f)) / (np.max(x1f) - np.min(x1f))
xmesh,ymesh=np.meshgrid(x3f,x1f)

#PLOTS


rho_theta = rho[0, :, ::-1] # different rho_theta for different phis, it is the same for different r
rho_r = rho[0, :, ::-1].T # different rho_r for different thetas, it is the same for different phis

#plt.plot(rho_r, r)
#plt.show()

plots, axes = plt.subplots(figsize=(8,6),subplot_kw=dict(projection='polar'))

plt.xlabel('$\\phi$')
plt.ylabel('$ r$')

#Density:
minval=min(rhoslice[0])
maxval=max(rhoslice[0])

norm=mpl.colors.Normalize(vmin=minval, vmax=maxval)
#norm=matplotlib.colors.LogNorm()
im=axes.pcolormesh(xmesh,ymesh,np.transpose(rhoslice),cmap=cmap1, norm=norm)   	

cbar=plots.colorbar(im, ax=axes)

# Turn off tick labels
axes.set_yticklabels([])
axes.set_xticklabels([])

plt.figtext(0.5, 0.01, 'time = '+str(time), ha="center", fontsize=12)
plt.title('Density')
axes.set_aspect('auto')
plt.show()


#rho in theta
plots, axes = plt.subplots(figsize=(8,6))

plt.xlabel('$\\phi$')
plt.ylabel('$ r$')

#Density:
minval=min(rho[0,:,:rmaxpos][0])
maxval=max(rho[0,:,:rmaxpos][0])

xmesh,ymesh=np.meshgrid(x2f,x1f)

#norm=mpl.colors.Normalize(vmin=minval, vmax=maxval)
norm=matplotlib.colors.LogNorm()
im=axes.pcolormesh(xmesh,ymesh,np.transpose(rho[0,:,:rmaxpos]),cmap=cmap1, norm=norm)     

cbar=plots.colorbar(im, ax=axes)

# Turn off tick labels
axes.set_yticklabels([])
axes.set_xticklabels([])

plt.figtext(0.5, 0.01, 'time = '+str(time), ha="center", fontsize=12)
plt.title('Density in theta')
axes.set_aspect('auto')
#plt.style.use('dark_background')
plt.show()


#The cut around a planet
plots, axes = plt.subplots(figsize=(8,6))

plt.xlabel('$\\phi$')
plt.ylabel('$ r$')
minval=min(rhoslice[0])
maxval=max(rhoslice[0])

norm=mpl.colors.Normalize(vmin=minval, vmax=maxval)
index = np.where((r>0.9) & (r<1.1))

xmesh,ymesh=np.meshgrid(x3f,x1f[index])
im=axes.pcolormesh(xmesh,ymesh,np.transpose(rhoslice)[index],cmap=cmap1, norm=norm)     

cbar=plots.colorbar(im, ax=axes)

plt.figtext(0.5, 0.01, 'time = '+str(time), ha="center", fontsize=12)
plt.title('In the vicinity of the planet')
axes.set_aspect('auto')
#plt.style.use('dark_background')
plt.show()


#Velocity:
xmesh,ymesh=np.meshgrid(x3f,x1f)
plots, axes = plt.subplots(figsize=(8,6),subplot_kw=dict(projection='polar'))
minval=min(velr_slice[0])
maxval=max(velr_slice[0])

norm=mpl.colors.Normalize(vmin=minval, vmax=maxval/10)
#norm=matplotlib.colors.LogNorm()
im=axes.pcolormesh(xmesh,ymesh,np.transpose(velr_slice),norm=norm,cmap=cmap2)    

cbar=plots.colorbar(im, ax=axes)

# Turn off tick labels
axes.set_yticklabels([])
axes.set_xticklabels([])

plt.figtext(0.5, 0.01, 'time = '+str(time), ha="center", fontsize=12)
plt.title('Radial velocity')
axes.set_aspect('auto')
plt.show()


plots, axes = plt.subplots(figsize=(8,6),subplot_kw=dict(projection='polar'))



#axes.streamplot(xmesh,ymesh,np.transpose(velphi_slice)[0],np.transpose(velphi_slice)[:][0], density=[0.5, 1])

minval=min(velphi_slice[0])
maxval=max(velphi_slice[0])

norm=mpl.colors.Normalize(vmin=minval, vmax=maxval)
im=axes.pcolormesh(xmesh,ymesh,np.transpose(velphi_slice),norm=norm,cmap=cmap1)    

cbar=plots.colorbar(im, ax=axes)

# Turn off tick labels
axes.set_yticklabels([])
axes.set_xticklabels([])



plt.figtext(0.5, 0.01, 'time = '+str(time), ha="center", fontsize=12)
plt.title('Azimuthal velocity')
axes.set_aspect('auto')
plt.show()



plots, axes = plt.subplots(figsize=(8,6),subplot_kw=dict(projection='polar'))

plt.xlabel('$\\phi$')
plt.ylabel('$ r$')

#Density:
minval=min(tempslice[0])
maxval=max(tempslice[0])

norm=mpl.colors.Normalize(vmin=minval, vmax=maxval)
#norm=matplotlib.colors.LogNorm()
im=axes.pcolormesh(xmesh,ymesh,np.transpose(tempslice),cmap=cmap1, norm=norm)    

cbar=plots.colorbar(im, ax=axes)

# Turn off tick labels
axes.set_yticklabels([])
axes.set_xticklabels([])

plt.figtext(0.5, 0.01, 'time = '+str(time), ha="center", fontsize=12)
plt.title('Temperature')
axes.set_aspect('auto')
plt.show()