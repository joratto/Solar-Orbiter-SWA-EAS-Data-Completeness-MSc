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


'''CDF configuration'''
cdf_filename_eas = 'solo_L1_swa-eas-padc_20231105T172733-20231105T174232_V01.cdf'
cdf_filename_mag = 'solo_L2_mag-srf-normal_20231105_V01.cdf'


'''CDF preparation'''
cdf_eas = pycdf.CDF('data\\' + cdf_filename_eas)
#print(cdf_eas)
time_eas = np.array(cdf_eas['EPOCH'])

cdf_mag = pycdf.CDF('data\\' + cdf_filename_mag)
#print(cdf_mag)
time_mag = cdf_mag['EPOCH']#[500000:])

t0_eas = time_eas[0] # start of B EAS timeframe
t0_mag = time_mag[0] # start of B MAG timeframe
tf_eas = time_eas[-1] # end of B EAS timeframe
tf_mag = time_mag[-1] # end of B MAG timeframe
print('EAS time series from {} to {}'.format(t0_eas, tf_eas))
print('MAG time series from {} to {}'.format(t0_mag, tf_mag))


'''transform matrices'''
r2o2 = np.sqrt(2)/2 # root 2 over 2
SRFtoEAS1 = np.array([[0,0,-1],[-r2o2,r2o2,0],[r2o2,r2o2,0]])
SRFtoEAS2 = np.array([[0,0,1],[-r2o2,-r2o2,0],[r2o2,-r2o2,0]])
SRFtoEASX = (SRFtoEAS1,SRFtoEAS2) # the transform matrices for SRF to the respective EAS head coordinates. First is EAS1, second is EAS2.

EAS1toSRF = np.array([[0,-r2o2,r2o2],[0,r2o2,r2o2],[-1,0,0]])
EAS2toSRF = np.array([[0,-r2o2,r2o2],[0,-r2o2,-r2o2],[1,0,0]])
EASXtoSRF = (EAS1toSRF,EAS2toSRF) # the inverse transform matrices


'''time series preparation'''
B_eas = cdf_eas['SWA_EAS_MagDataUsed'] # onboard EAS B (SRF)
B_mag = cdf_mag['B_SRF'] # reported MAG B (SRF)

B_mag, time_mag, B_eas, time_eas = fx.cropTimeToOverlap(B_mag, time_mag, B_eas, time_eas, searchdivisions=5)

B_eas_magnitude = np.ndarray((len(B_eas))) # (SRF)
B_mag_magnitude = np.ndarray((len(B_mag))) # (SRF)

B_angle = np.ndarray((len(B_mag)))

eas_used = cdf_eas['SWA_EAS_EasUsed']
B_eas_EASX = np.ndarray((len(B_eas),3)) # (EASX)
B_mag_EASX = np.ndarray((len(B_mag),3)) # (EASX)

B_eas_spherical_SRF = np.ndarray((len(B_eas),3)) # (SRF)
B_mag_spherical_SRF = np.ndarray((len(B_mag),3)) # (SRF)

B_eas_spherical_EASX = np.ndarray((len(B_eas),3)) # (EASX)
B_mag_spherical_EASX = np.ndarray((len(B_mag),3)) # (EASX)

B_eas_elevation_used_parallel, B_eas_elevation_used_antiparallel = np.array(cdf_eas['SWA_EAS_ELEVATION']).T

for i in range(len(B_mag)):
    # calculate vectors and magnitudes:
    vector_mag = B_mag[i]
    vector_mag_magnitude = np.linalg.norm(vector_mag)
    B_mag_magnitude[i] = vector_mag_magnitude

    vector_eas = B_eas[i]
    vector_eas_magnitude = np.linalg.norm(vector_eas)
    B_eas_magnitude[i] = vector_eas_magnitude

    # normalise B_mag:
    vector_mag = vector_mag/vector_mag_magnitude #!!!
    B_mag[i] = vector_mag

    # calculate angles between MAG vectors and EAS vectors:
    B_angle[i] = np.arccos(np.dot(vector_mag,vector_eas))*180/np.pi

    # transform from SRF to respective EAS head coordinates:
    vector_mag_magnitude_EASX = SRFtoEASX[eas_used[i]].dot(vector_mag)
    vector_eas_magnitude_EASX = SRFtoEASX[eas_used[i]].dot(vector_eas)
    B_mag_EASX[i] = vector_mag_magnitude_EASX
    B_eas_EASX[i] = vector_eas_magnitude_EASX

    # transform SRF and EASX to respective spherical coordinates w/elevation and azimuth:
    B_mag_spherical_SRF[i] = fx.cartToSphere(vector_mag)
    B_mag_spherical_EASX[i] = fx.cartToSphere(vector_mag_magnitude_EASX)
    B_eas_spherical_SRF[i] = fx.cartToSphere(vector_eas)
    B_eas_spherical_EASX[i] = fx.cartToSphere(vector_eas_magnitude_EASX)


'''plot configuration'''
geometry = 'spherical' # 'cartesian', 'spherical'
coordinates = 'EASX' # 'SRF','EASX'

Nplots = 4 # number of plots to show
sns.set_theme(style='ticks')
fig1, axs = plt.subplots(Nplots)

axs[0].set_title('{}    &    {}'.format(cdf_filename_eas, cdf_filename_mag))


'''coordinates options'''
coordinates_dictionary = {'SRF':{'cartesian':(B_mag,B_eas),'spherical':(B_mag_spherical_SRF,B_eas_spherical_SRF)}, 
                        'EASX':{'cartesian':(B_mag_EASX,B_eas_EASX),'spherical':(B_mag_spherical_EASX,B_eas_spherical_EASX)}}
Bx_mag, By_mag, Bz_mag = np.array(coordinates_dictionary[coordinates][geometry][0]).T
Bx_eas, By_eas, Bz_eas = np.array(coordinates_dictionary[coordinates][geometry][1]).T


'''time delay stuff'''
sr = 8
delayBx, lagsBx, corrBx = fx.delayFinder(np.pad(Bx_mag,(0,0)),Bx_eas[:],sr)
delayBy, lagsBy, corrBy = fx.delayFinder(np.pad(By_mag,(0,0)),By_eas[:],sr)
delayBz, lagsBz, corrBz = fx.delayFinder(np.pad(Bz_mag,(0,0)),Bz_eas[:],sr)

for i in range(len(corrBx)):
    corrBx[i] = np.linalg.norm(corrBx[i])
    corrBy[i] = np.linalg.norm(corrBy[i])
    corrBz[i] = np.linalg.norm(corrBz[i])


'''plots'''
# unit BR comparison
ax = axs[0]
ax.plot(time_mag,Bx_mag)
ax.plot(time_eas[:-1],Bx_eas[:-1])
ax.set_ylabel(r'$B_{r}$'+'\n(unit {})'.format(coordinates))
ax.legend([r"$B_{MAG,r}$", r"$B_{EAS,r}$"])
ax.set_ylim(-0.2,2.2)
ax.grid()

# unit Btheta comparison
ax = axs[1]
ax.plot(time_mag,By_mag)
ax.plot(time_eas[:-1],By_eas[:-1])
ax.plot(time_eas[:-1],B_eas_elevation_used_parallel[:-1])
ax.plot(time_eas[:-1],B_eas_elevation_used_antiparallel[:-1])
ax.set_ylabel(r'$B_{θ}$'+'\n(degrees {})'.format(coordinates))
ax.legend([r"$B_{MAG,θ}$", r"$B_{EAS,θ}$", r"$B_{EAS:Used,θ↑↑}$", r"$B_{EAS:Used,θ↑↓}$"])
ax.set_ylim(-55,55)
tick_spacing = 15 # degrees
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.grid()

# unit Bphi comparison
ax = axs[2]
ax.plot(time_mag,Bz_mag)
ax.plot(time_eas[:-1],Bz_eas[:-1])
ax.set_ylabel(r'$B_{φ}$'+'\n(degrees {})'.format(coordinates))
ax.legend([r"$B_{MAG,φ}$", r"$B_{EAS,φ}$"])
ax.set_ylim(-15,375)
tick_spacing = 45 # degrees
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.grid()

# EAS sensor head used
ax = axs[3]
ax.plot(time_eas,eas_used,color='green')
ax.set_ylabel('Sensor Used')
ax.set_ylim(-0.2,1.2)
ax.set_yticks([0,1],['EAS1','EAS2'])
ax.grid()


'''
# unit Bx comparison
ax = axs[0]
ax.plot(time_mag,Bx_mag)
ax.plot(time_eas,Bx_eas)
ax.set_ylabel(r'$B_{x}$'+'\n(unit {})'.format(coordinates))
ax.legend([r"$B_{MAG,x}$", r"$B_{EAS,x}$"])
ax.set_ylim(-1.2,1.2)
ax.grid()

# unit By comparison
ax = axs[1]
ax.plot(time_mag,By_mag)
ax.plot(time_eas,By_eas)
ax.set_ylabel(r'$B_{y}$'+'\n(unit {})'.format(coordinates))
ax.legend([r"$B_{MAG,y}$", r"$B_{EAS,y}$"])
ax.set_ylim(-1.2,1.2)
ax.grid()

# unit Bz comparison
ax = axs[2]
ax.plot(time_mag,Bz_mag)
ax.plot(time_eas,Bz_eas)
ax.set_ylabel(r'$B_{z}$'+'\n(unit {})'.format(coordinates))
ax.legend([r"$B_{MAG,z}$", r"$B_{EAS,z}$"])
ax.set_ylim(-1.2,1.2)
ax.grid()

# EAS sensor head used
ax = axs[3]
ax.plot(time_eas,eas_used,color='green')
ax.set_ylabel('Sensor Used')
ax.set_ylim(-0.2,1.2)
ax.set_yticks([0,1],['EAS1','EAS2'])
ax.grid()
'''

'''
# EAS-MAG B-vector angle difference
ax = axs[3]
ax.plot(time_mag,B_angle, color='green')
#ax.plot(time_eas[:-1],B_angle, color='red') # in case you want to compare the two same series! (one is  currently one point shorter, for some reason)
ax.set_ylabel('angle between\nB vectors (°)')
#tick_spacing = 2 # degrees
#ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.legend([r'$B_{MAG}-B_{EAS}$ Angle'])
ax.grid()

axs[Nplots-1].set_xlabel('Date & Time\n(dd hh:mm)')#axs[Nplots-1].set_xlabel('Date & Time (dd hh:mm)')
'''

'''
# unit B comparison (all axes)
ax = axs[0]
ax.plot(time_mag,Bx_mag)
ax.plot(time_eas,Bx_eas)
ax.plot(time_mag,By_mag)
ax.plot(time_eas,By_eas)
ax.plot(time_mag,Bz_mag)
ax.plot(time_eas,Bz_eas)
ax.set_ylabel('B\n(normalised)')
ax.legend(["Bx MAG", "Bx EAS","By MAG", "By EAS","Bz MAG", "Bz EAS"])
ax.set_ylim(-1.2,1.2)
ax.grid()
'''

'''
# B magnitude comparison
ax = axs[3]
ax.plot(time_mag,B_mag_magnitude)
ax.plot(time_eas,B_eas_magnitude)
ax.set_ylabel('|B|\n(nT)')
ax.legend(["|B| MAG", "|B| EAS"])
ax.set_ylim(-1.2,1.2)
ax.grid()
'''

'''
# B time delay comparison
ax = axs[Nplots-1]
#ax.plot(lagsBx,corrBx)
#ax.plot(lagsBy,corrBy)
#ax.plot(lagsBz,corrBz)
#ax.set_ylabel('correlation\n(dimensionless)')
#ax.set_xlabel('possible time delays (seconds)')
#ax.set_yticklabels([''])
#ax.legend(["Bx Correlation", "By Correlation", "Bz Correlation"])
#ax.grid()

plt.show(block=True) # set to block=False to automatically close figure
'''

ax = axs[0]

plt.show()