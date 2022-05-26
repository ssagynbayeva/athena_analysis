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
#rcParams['text.usetex']=True
#rcParams['text.latex.unicode']=True


#matplotlib.rc('font', family='serif', serif='cm10')
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black','cyan','magenta'])

# Athena++ modules
import athena_read

directory = '/Users/sabina/Desktop/work/seawulf/diff_SMR3'

leveln = None
quantities=['rho','vel1','vel2','vel3']
for filename in os.listdir(directory):
    if filename.endswith('.athdf'):

        input_filename = filename

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

        x1f=x1f[:rmaxpos+1]

        #logr=np.log10(x1f)
        #x1f_norm = (x1f - np.min(x1f)) / (np.max(x1f) - np.min(x1f))
        xmesh,ymesh=np.meshgrid(x3f,x1f)

        #PLOTS


        rho_theta = rho[0, :, ::-1] # different rho_theta for different phis, it is the same for different r
        rho_r = rho[0, :, ::-1].T # different rho_r for different thetas, it is the same for different phis

        '''

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


        if time<10:
            tiiime = '0'+str(round(time,2))
        elif time<100:
            tiiime = str(round(time,2))
        plt.savefig('den_m=1.5e-5_time_'+tiiime+'.png')
        '''

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
        if time<10:
            tiiime = '0'+str(round(time,2))
        elif time<100:
            tiiime = str(round(time,2))
        plt.savefig('zoomed_den_m=1.5e-5_time_'+tiiime+'.png')



exec(open("make_animations.py").read())
