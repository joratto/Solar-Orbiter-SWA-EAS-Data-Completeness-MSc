import numpy as np
import scipy as sp
import astropy as astro
import astropy.coordinates as astrocoo
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
os.environ["CDF_LIB"] = "C:\\Program Files\\CDF_Distribution"
from spacepy import pycdf
import datetime
import pandas as pd

import functionsMSc as fx
from useful_Matrices import *


'''bin projection'''
Nplots = 1 # number of plots to show
#sns.set_theme(style='darkgrid')
fig, ax = plt.subplots(Nplots)

point_density = 1
headToProject = 0
bin_bounds = fx.getEASXBinProjections(headToProject,bin_dictionary,EASXtoSRF,point_density=point_density)
for elev in bin_bounds:
        for azim in elev:
                ax.scatter(azim[2],azim[1],color='blue',marker='.',s=0.1)

# headToProject = 1
# bin_bounds = fx.getEASXBinProjections(headToProject,bin_dictionary,EASXtoSRF,point_density=1)
# for elev in bin_bounds:
#         for azim in elev:
#                 ax.scatter(azim[2],azim[1],color='red',marker='.',s=0.1)

time = datetime.datetime(2023,11,5,hour=17,minute=29,second=38)

vectorEAS = np.array([1, 39.611968353947965, 294.52237316250864])
vectorEAS[2] = (vectorEAS[2]) - 360*(vectorEAS[2] > 180)
vectorMAG = np.array([1, 36.07976192980311, 293.0752082175452])
vectorMAG[2] = (vectorMAG[2]) - 360*(vectorMAG[2] > 180)

vectorEAS, antivectorEAS, contourArrayEAS = fx.getEASXVectorProjections(headToProject,vector=vectorEAS,pitch_contours=[5,15,30,60,90,120,150,165,175],cart_proj_tuple=EASXtoSRF,point_density=point_density)
vectorMAG, antivectorMAG, contourArrayMAG = fx.getEASXVectorProjections(headToProject,vector=vectorMAG,pitch_contours=[5,15,30,60,90,120,150,165,175],cart_proj_tuple=EASXtoSRF,point_density=point_density)

contourRadiusEAS, contourElevEAS, contourAzimEAS = contourArrayEAS
contourRadiusMAG, contourElevMAG, contourAzimMAG = contourArrayMAG

ax.scatter(contourAzimEAS,contourElevEAS,color='purple',marker='.',s=0.1)
ax.scatter(contourAzimMAG,contourElevMAG,color='green',marker='.',s=0.1)

# shaped markers
markersize = 15
ax.plot(vectorEAS[2],vectorEAS[1],markerfacecolor='none',markeredgecolor='purple',marker='D',markeredgewidth=2,markersize=markersize)
ax.plot(antivectorEAS[2],antivectorEAS[1],markerfacecolor='none',markeredgecolor='purple',marker='s',markeredgewidth=2,markersize=markersize)
ax.plot(vectorMAG[2],vectorMAG[1],markerfacecolor='none',markeredgecolor='green',marker='D',markeredgewidth=2,markersize=markersize)
ax.plot(antivectorMAG[2],antivectorMAG[1],markerfacecolor='none',markeredgecolor='green',marker='s',markeredgewidth=2,markersize=markersize)

# point markers
pointsize = 2
ax.plot(vectorEAS[2],vectorEAS[1],color='purple',marker='.',markeredgewidth=2,markersize=pointsize)
ax.plot(antivectorEAS[2],antivectorEAS[1],color='purple',marker='.',markeredgewidth=2,markersize=pointsize)
ax.plot(vectorMAG[2],vectorMAG[1],color='green',marker='.',markeredgewidth=2,markersize=pointsize)
ax.plot(antivectorMAG[2],antivectorMAG[1],color='green',marker='.',markeredgewidth=2,markersize=pointsize)


ax.set_title('EAS{}, {}'.format(headToProject+1, time))

ax.set_ylim(-90,90)
ax.set_xlim(-180,180)
elev_tick_spacing = 15 # degrees
ax.yaxis.set_major_locator(ticker.MultipleLocator(elev_tick_spacing))
azim_tick_spacing = 30 # degrees
ax.xaxis.set_major_locator(ticker.MultipleLocator(azim_tick_spacing))
ax.set_ylabel('Elevation (degrees SRF)')
ax.set_xlabel('Azimuth (degrees SRF)')
ax.set_aspect(1)
#ax.grid()

plt.show()