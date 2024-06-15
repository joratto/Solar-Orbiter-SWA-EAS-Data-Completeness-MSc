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

import functionsMSc as fx
from useful_Matrices import *

'''bin projection'''
Nplots = 1 # number of plots to show
sns.set_theme(style='darkgrid')
fig, ax = plt.subplots(Nplots)

headToProject = 0
bin_bounds = fx.getEASXBinProjections(headToProject,bin_dictionary,EASXtoSRF,point_density=1)
for elev in bin_bounds:
        for azim in elev:
                ax.scatter(azim[2],azim[1],color='blue',marker='.',s=0.1)

# headToProject = 1
# bin_bounds = fx.getEASXBinProjections(headToProject,bin_dictionary,EASXtoSRF,point_density=10)
# for elev in bin_bounds:
#         for azim in elev:
#                 ax2.scatter(azim[2],azim[1],color='red',marker='.',s=0.1)

ax.set_ylim(-90,90)
ax.set_xlim(-180,180)
elev_tick_spacing = 15 # degrees
ax.yaxis.set_major_locator(ticker.MultipleLocator(elev_tick_spacing))
azim_tick_spacing = 30 # degrees
ax.xaxis.set_major_locator(ticker.MultipleLocator(azim_tick_spacing))
ax.set_ylabel('Elevation (degrees SRF)')
ax.set_xlabel('Azimuth (degrees SRF)')
ax.set_aspect(1)

plt.show()