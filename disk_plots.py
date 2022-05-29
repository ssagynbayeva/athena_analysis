"""
plot global accretion disks

usage: disk_plots.py [-h] [-c NUM_CORES] [--dir SAVE_DIRECTORY]
                              [--hide] [-v VERSION] [--range R_LIM R_LIM]
                              [--shift] [--cmap CMAP] [--cmax CMAX]
                              [--fontsize FONTSIZE] [--dpi DPI]
                              frames [frames ...]

positional arguments:
  frames                select single frame or range(start, end, rate). error
                        if nargs != 1 or 3

optional arguments:
  -h, --help            show this help message and exit
  -c NUM_CORES          number of cores (default: 1)
  --dir SAVE_DIRECTORY  save directory (default: dustDensityMaps)
  --hide                for single plot, do not display plot (default: display
                        plot)
  -v VERSION            version number (up to 4 digits) for this set of plot
                        parameters (default: None)
  --range R_LIM R_LIM   radial range in plot (default: [r_min, r_max])
  --shift               center frame on vortex peak or middle (default: do not
                        center)
  --cmap CMAP           color map (default: viridis)
  --cmax CMAX           maximum density in colorbar (default: 10 for hcm+, 2.5
                        otherwise)
  --fontsize FONTSIZE   fontsize of plot annotations (default: 16)
  --dpi DPI             dpi of plot annotations (default: 100)
"""
import sys
sys.path.insert(0, '$ATHENA_DIR/vis/python')
import pickle, glob
from multiprocessing import Pool
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pylab import *
import struct
import array
import os
from scipy.interpolate import griddata
import h5py

# Athena++ modules
import athena_read

cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ['black','cyan','magenta'])
cmap2 = 'PuRd'

###############################################################################

### Input Parameters ###

def new_argument_parser(description = "Plot global accretion disks."):
    parser = argparse.ArgumentParser()

    # Files
    parser.add_argument('--num', dest = "filenum", default = 10,
                         help = 'file number (default: 10)')

    parser.add_argument('--dir', dest = "save_directory", default = None,
                         help = 'save directory (default: None)')

    parser.add_argument('--file', dest = "filename", default = None,
                         help = 'the input file name (default: None)')

    # Quantity to plot
    parser.add_argument('--den', dest = "den", action = 'store_true', default = False,
                         help = 'plot density (default: False)')

    parser.add_argument('--az_vel', dest = "az_vel", action = 'store_true', default = False,
                         help = 'plot azimuthal velocity (default: False)')

    parser.add_argument('--rad_vel', dest = "rad_vel", action = 'store_true', default = False,
                         help = 'plot radial velocity (default: False)')

    # Plot Parameters (variable)
    parser.add_argument('--hide', dest = "show", action = 'store_false', default = True,
                         help = 'for single plot, do not display plot (default: display plot)')
    parser.add_argument('-v', dest = "version", type = int, default = None,
                         help = 'version number (up to 4 digits) for this set of plot parameters (default: None)')

    parser.add_argument('--range', dest = "r_lim", type = float, nargs = 2, default = None,
                         help = 'radial range in plot (default: [r_min, r_max])')
    parser.add_argument('--shift', dest = "center", action = 'store_true', default = False,
                         help = 'center frame on vortex peak or middle (default: do not center)')

    # Plot Parameters (contours)
    parser.add_argument('--contour', dest = "use_contours", action = 'store_true', default = False,
                         help = 'use contours or not (default: do not use contours)')
    parser.add_argument('--low', dest = "low_contour", type = float, default = 1.1,
                         help = 'lowest contour (default: 1.1)')
    parser.add_argument('--high', dest = "high_contour", type = float, default = 3.5,
                         help = 'highest contour (default: 3.5)')
    parser.add_argument('--num_levels', dest = "num_levels", type = int, default = None,
                         help = 'number of contours (choose this or separation) (default: None)')
    parser.add_argument('--separation', dest = "separation", type = float, default = 0.1,
                         help = 'separation between contours (choose this or num_levels) (default: 0.1)')

    # Plot Parameters (quiver)
    parser.add_argument('--quiver', dest = "quiver", action = 'store_true', default = False,
                         help = 'use velocity quivers or not (default: do not use quivers)')
    parser.add_argument('--start', dest = "start_quiver", type = float, default = None,
                         help = 'start of quiver region in radius (default: r_lim[0])')
    parser.add_argument('--end', dest = "end_quiver", type = float, default = None,
                         help = 'end of quiver region in radius (default: r_lim[1])')
    parser.add_argument('--rate_x', dest = "quiver_rate_x", type = int, default = 6,
                         help = 'sub_sample in radius (default: 6)')
    parser.add_argument('--rate_y', dest = "quiver_rate_y", type = int, default = 100,
                         help = 'sub_sample in angle (default: 24)')
    parser.add_argument('--scale', dest = "quiver_scale", type = float, default = 0.25,
                         help = 'bigger scale means smaller arrow (default: 1)')
    
    # Plot Parameters (rarely need to change)
    parser.add_argument('--cmap', dest = "cmap", default = cmap1,
                         help = 'color map (default: cmap1)')
    parser.add_argument('--crange', dest = "c_lim", type = float, nargs = 2, default = None,
                         help = 'range in colorbar (default: [-0.2, 0])')

    parser.add_argument('--fontsize', dest = "fontsize", type = int, default = 16,
                         help = 'fontsize of plot annotations (default: 16)')
    parser.add_argument('--dpi', dest = "dpi", type = int, default = 100,
                         help = 'dpi of plot annotations (default: 100)')

    return parser

###############################################################################

### Parse Arguments ###
args = new_argument_parser().parse_args()

## Get Athena++ parameters ##
if int(args.filenum) < 10:
    input_filename = 'disk.out1.0000'+str(args.filenum)+'.athdf'
else:
    input_filename = 'disk.out1.000'+str(args.filenum)+'.athdf'
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


### Get Input Parameters ###

# Files
#save_directory = args.save_directory
#if not os.path.isdir(save_directory):
#    os.mkdir(save_directory) # make save directory if it does not already exist

# Quantity to Plot
density = args.den
azimuthal_velocity = args.az_vel
radial_velocity = args.rad_vel


# Plot Parameters (variable)
show = args.show


# Plot Parameters (contours)
use_contours = args.use_contours
low_contour = args.low_contour
high_contour = args.high_contour
num_levels = args.num_levels
if num_levels is None:
    separation = args.separation
    num_levels = int(round((high_contour - low_contour) / separation + 1, 0))

# Plot Parameters (quiver)
quiver = args.quiver
start_quiver = args.start_quiver
end_quiver = args.end_quiver
rate_x = args.quiver_rate_x
rate_y = args.quiver_rate_y
scale = args.quiver_scale
#x_min = min(xmesh) #for now, should change
#x_max = max(xmesh)
'''
if start_quiver is None:
   start_quiver = x_min
if end_quiver is None:
   end_quiver = x_max
'''
# Plot Parameters (constant)
cmap = args.cmap
fontsize = args.fontsize
dpi = args.dpi

##### PLOTTING #####

def make_plot(show = True):

    ### Plot ###

    plots, axes = plt.subplots(figsize=(8,6),subplot_kw=dict(projection='polar'), dpi=dpi)

    # Density:
    if density:
      minval=min(rhoslice[0])
      maxval=max(rhoslice[0])

      norm=mpl.colors.Normalize(vmin=minval, vmax=maxval)
      #norm=matplotlib.colors.LogNorm()
      im=axes.pcolormesh(xmesh,ymesh,np.transpose(rhoslice),cmap=cmap1, norm=norm)    

      cbar=plots.colorbar(im, ax=axes)
      plt.title('Density')

      # Radial velocity:
    if args.rad_vel:
      minval=min(velr_slice[0])
      maxval=max(velr_slice[0])

      norm=mpl.colors.Normalize(vmin=minval, vmax=maxval/10)
      #norm=matplotlib.colors.LogNorm()
      im=axes.pcolormesh(xmesh,ymesh,np.transpose(velr_slice),norm=norm,cmap=cmap1)    

      cbar=plots.colorbar(im, ax=axes)
      plt.title('Radial velocity')

    if use_contours:
        levels = np.linspace(low_contour, high_contour, num_levels)
        colors = generate_colors(num_levels)
        plot.contour(x, y, np.transpose(normalized_density), levels = levels, origin = 'upper', linewidths = 1, colors = colors, alpha = 0.8)

    if quiver:
        # Velocity
        radial_velocity = vrad
        azimuthal_velocity = vtheta
        keplerian_velocity = rad * (np.power(rad, -1.5) - 1)
        azimuthal_velocity -= keplerian_velocity[:, None]

        if center:
            radial_velocity = np.roll(radial_velocity, shift_c, axis = -1)
            azimuthal_velocity = np.roll(azimuthal_velocity, shift_c, axis = -1)

        # Sub-sample the grid (i.e. Only plot arrows over a range of radii instead of the entire grid.)
        start_i = np.searchsorted(rad, start_quiver)
        end_i = np.searchsorted(rad, end_quiver)

        x_q = x[start_i:end_i]
        y_q = y[:]
        u = np.transpose(radial_velocity)[:, start_i:end_i]
        v = np.transpose(azimuthal_velocity)[:, start_i:end_i]

        # plot.quiver(x values aka the radial coordinates, y values aka the azimuthal coordinates, corresponding radial velocity, corresponding azimuthal velocity)
        plot.quiver(x_q[::rate_x], y_q[::rate_y], u[::rate_y,::rate_x], v[::rate_y,::rate_x], scale = scale) ## This is it really
        # Scale sets the size of the arrows (Look up the documentation to see what a scale of "1" means. I think it is literally \delta R = "1" or \delta \Phi = "1" on the grid)
        # rate_x and rate_y determine how often to plot the arrows (once every rate_x in the radial direction, and once every rate_y in the azimuthal direction.)
        # You probably don't want to plot an arrow for every cell. The "::rate_x" is a weird Python trick I don't actually understand, but x[::5] plots x[0, 5, 10, 15, 20, etc.]

    # Axes
    # Turn off tick labels
    axes.set_yticklabels([])
    axes.set_xticklabels([])


    plt.xlabel(r"r", fontsize = fontsize)
    plt.ylabel(r"$\phi$", fontsize = fontsize)

    plt.figtext(0.5, 0.01, 'time = '+str(time), ha="center", fontsize=fontsize)
    axes.set_aspect('auto')
    #plt.style.use('dark_background')
    plt.show()


    plt.close(fig) # Close Figure (to avoid too many figures)

make_plot(show=True)
