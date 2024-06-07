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
cdf_filename_eas = 'solo_L1_swa-eas-padc_20231105T172733-20231105T174232_V01' + '.cdf'
cdf_filename_mag = 'solo_L2_mag-srf-normal_20231105_V01' + '.cdf'


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


'''useful matrices'''
r2o2 = np.sqrt(2)/2 # root 2 over 2
SRFtoEAS1 = np.array([[0,0,-1],[-r2o2,r2o2,0],[r2o2,r2o2,0]])
SRFtoEAS2 = np.array([[0,0,1],[-r2o2,-r2o2,0],[r2o2,-r2o2,0]])
SRFtoEASX = (SRFtoEAS1,SRFtoEAS2) # the transform matrices for SRF to the respective EAS head coordinates

EAS1toSRF = np.array([[0,-r2o2,r2o2],[0,r2o2,r2o2],[-1,0,0]])
EAS2toSRF = np.array([[0,-r2o2,r2o2],[0,-r2o2,-r2o2],[1,0,0]])
EASXtoSRF = (EAS1toSRF,EAS2toSRF) # the inverse transform matrices

EAS1z_SRF = EAS1toSRF.T[2] # the EASx z axis in the SRF frame is equivalent to the third column of EASxtoSRF
EAS2z_SRF = EAS2toSRF.T[2]
EASXz_SRF = (EAS1z_SRF, EAS2z_SRF)
print('\nEAS1z SRF = {}\nEAS2z SRF = {}'.format(EAS1z_SRF, EAS2z_SRF))


'''bin dictionaries'''
EAS1_bin_dict = {'ELEVATION':np.array([39.34,  29.17,  20.91,  13.98,   8.06,   2.91,  -1.66,  -5.82,  -9.7, -13.43, -17.13, -20.94, -25., -29.53, -34.82, -41.36]),
        'ELEVATION_delta_lower':np.array([5.66,  4.514, 3.748, 3.179, 2.74,  2.409, 2.161, 1.996, 1.886, 1.841, 1.856, 1.95, 2.115, 2.413, 2.88, 3.655]),
        'ELEVATION_delta_upper':np.array([5.66,  4.514, 3.748, 3.179, 2.74,  2.409, 2.161, 1.996, 1.886, 1.841, 1.856, 1.95, 2.115, 2.413, 2.88, 3.655]),
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}
EAS1_bin_dict['ELEVATION_lower_bound'] = EAS1_bin_dict['ELEVATION'] - EAS1_bin_dict['ELEVATION_delta_lower'] # subtract elevation bin lower deltas from elevation bin centers to get the lower bounds
EAS1_bin_dict['AZIMUTH_lower_bound'] = EAS1_bin_dict['AZIMUTH'] - EAS1_bin_dict['AZIMUTH_delta_lower']
EAS1_bin_dict['ELEVATION_upper_bound'] = EAS1_bin_dict['ELEVATION'] + EAS1_bin_dict['ELEVATION_delta_upper'] # add elevation bin upper deltas to elevation bin centers to get the upper bounds
EAS1_bin_dict['AZIMUTH_upper_bound'] = EAS1_bin_dict['AZIMUTH'] + EAS1_bin_dict['AZIMUTH_delta_upper']

EAS2_bin_dict = {'ELEVATION':np.array([38.94,  28.25,  19.86,  12.99,   7.25,   2.35,  -1.93,  -5.78,  -9.37, -12.84, -16.32, -19.97, -23.97, -28.57, -34.13, -41.1]),
        'ELEVATION_delta_lower':np.array([6.06,  4.633, 3.761, 3.111, 2.624, 2.272, 2.012, 1.838, 1.747, 1.722, 1.759, 1.887, 2.113, 2.485, 3.071, 3.897]),
        'ELEVATION_delta_upper':np.array([6.06,  4.633, 3.761, 3.111, 2.624, 2.272, 2.012, 1.838, 1.747, 1.722, 1.759, 1.887, 2.113, 2.485, 3.071, 3.897]),
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}
EAS2_bin_dict['ELEVATION_lower_bound'] = EAS2_bin_dict['ELEVATION'] - EAS2_bin_dict['ELEVATION_delta_lower'] # subtract elevation bin lower deltas from elevation bin centers to get the lower bounds
EAS2_bin_dict['AZIMUTH_lower_bound'] = EAS2_bin_dict['AZIMUTH'] - EAS2_bin_dict['AZIMUTH_delta_lower']
EAS2_bin_dict['ELEVATION_upper_bound'] = EAS2_bin_dict['ELEVATION'] + EAS2_bin_dict['ELEVATION_delta_upper'] # add elevation bin upper deltas to elevation bin centers to get the upper bounds
EAS2_bin_dict['AZIMUTH_upper_bound'] = EAS2_bin_dict['AZIMUTH'] + EAS2_bin_dict['AZIMUTH_delta_upper']

bin_dictionary = (EAS1_bin_dict, EAS2_bin_dict) # the bin data for the two EAS heads

#print(EAS1_bin_dict['ELEVATION_lower_bound'])
#print(EAS1_bin_dict['ELEVATION'])
#print(EAS1_bin_dict['ELEVATION_upper_bound'])
#print(EAS1_bin_dict['AZIMUTH_lower_bound'])
#print(EAS1_bin_dict['AZIMUTH'])
#print(EAS1_bin_dict['AZIMUTH_upper_bound'])
#print('HOOOLD IIIIT')


'''time series processing'''
# raw basic time series
B_eas = cdf_eas['SWA_EAS_MagDataUsed'] # onboard EAS B (SRF, cart)
B_mag = cdf_mag['B_SRF'] # reported MAG B (SRF, cart)

# processing basic time series
B_mag, time_mag, B_eas, time_eas = fx.cropTimeToOverlap(B_mag, time_mag, B_eas, time_eas, searchdivisions=5)

len_B_eas = len(B_eas)
len_B_mag = len(B_mag)
print('B_eas vector count = {}\nB_mag vector count = {}'.format(len_B_eas, len_B_mag))

# magnitude
B_eas_magnitude = np.ndarray((len_B_eas)) # (SRF)
B_mag_magnitude = np.ndarray((len_B_mag)) # (SRF)

# angles
B_angle = np.ndarray((len_B_mag)) # angle between B_eas SRF and B_mag SRF
B_eas_EAS1z_angle = np.ndarray((len_B_eas))
B_eas_EAS2z_angle = np.ndarray((len_B_eas))
B_eas_EASXz_angle = (B_eas_EAS1z_angle, B_eas_EAS2z_angle) # angles between B_eas SRF and EASXz SRF
B_mag_EAS1z_angle = np.ndarray((len_B_mag))
B_mag_EAS2z_angle = np.ndarray((len_B_mag))
B_mag_EASXz_angle = (B_mag_EAS1z_angle, B_mag_EAS2z_angle) # angles between B_mag SRF and EASXz SRF

# EAS used
eas_used = cdf_eas['SWA_EAS_EasUsed'] # the head used for B_eas in the actual data
B_eas_head = np.ndarray(len_B_eas, dtype=int) # the head that should've been used for B_eas, calculated
B_mag_head = np.ndarray(len_B_mag, dtype=int) # the head that should've been used for B_mag, calculated

# basic time series in EASX frame, using eas_used
B_eas_EASX = np.ndarray((len_B_eas,3)) # (EASX, cart)
B_mag_EASX = np.ndarray((len_B_mag,3)) # (EASX, cart)

# basic time series in EASX frame, using calculated head
B_eas_cart_EASX = np.ndarray((len_B_eas,3)) # (EASX, cart)
B_mag_cart_EASX = np.ndarray((len_B_mag,3)) # (EASX, cart)

# basic time series in SRF frame, spherical
B_eas_spherical_SRF = np.ndarray((len_B_eas,3)) # (SRF, sphere)
B_mag_spherical_SRF = np.ndarray((len_B_mag,3)) # (SRF, sphere))

# basic time series in EASX frame, spherical, using eas_used
B_eas_spherical_EASX = np.ndarray((len_B_eas,3)) # (EASX, sphere))
B_mag_spherical_EASX = np.ndarray((len_B_mag,3)) # (EASX, sphere))

# basic time series in EASX frame, spherical, using calculated head
B_eas_sphe_EASX = np.ndarray((len_B_eas,3)) # (EASX, sphere))
B_mag_sphe_EASX = np.ndarray((len_B_mag,3)) # (EASX, sphere))

# EASX elevation used in various frames and geometries
B_eas_elevation_used_parallel, B_eas_elevation_used_antiparallel = np.array(cdf_eas['SWA_EAS_ELEVATION']).T # (EASX, sphere)

# calculated EASX elevation bin indices (parallel and antiparallel)
B_eas_elev_bin = np.ndarray((len_B_eas,2))
B_mag_elev_bin = np.ndarray((len_B_mag,2))

# calculated EASX elevation (parallel and antiparallel)
B_eas_elev_EASX = np.ndarray((len_B_eas,2))
B_mag_elev_EASX = np.ndarray((len_B_mag,2))

# calculated EAS azimuth bin index (parallel and antiparallel)
B_eas_azim_bin = np.ndarray((len_B_eas,2))
B_mag_azim_bin = np.ndarray((len_B_eas,2))

# calculated EASX azimuth (parallel and antiparallel)
B_eas_azim_EASX = np.ndarray((len_B_eas,2))
B_mag_azim_EASX = np.ndarray((len_B_mag,2))

print('\n')
for i in range(len_B_mag):
    print('\rtime steps processed = {}/{}'.format(i+1,len_B_mag), end='')
    # calculate vectors and magnitudes:
    vector_mag = B_mag[i]
    vector_mag_magnitude = np.sqrt(vector_mag[0]**2+vector_mag[1]**2+vector_mag[2]**2)
    #vector_mag_magnitude = np.linalg.norm(vector_mag)
    B_mag_magnitude[i] = vector_mag_magnitude

    vector_eas = B_eas[i]
    vector_eas_magnitude = np.sqrt(vector_eas[0]**2+vector_eas[1]**2+vector_eas[2]**2)
    #vector_eas_magnitude = np.linalg.norm(vector_eas)
    B_eas_magnitude[i] = vector_eas_magnitude

    # normalise B_mag:
    vector_mag = vector_mag/vector_mag_magnitude #!!!
    B_mag[i] = vector_mag

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

    # calculate angles between MAG vectors and EAS vectors:
    B_angle[i] = np.arccos(np.dot(vector_mag,vector_eas))*180/np.pi

    # calculate EASX elevation and azimuth bins for EAS vectors:
    head = 0
    head, B_mag_EASXz_angle[0][i], B_mag_EASXz_angle[1][i] = fx.headPicker(vector_mag, EASXz_SRF[0], EASXz_SRF[1])
    B_mag_head[i] = head

    vector_mag_cart_EASX = SRFtoEASX[head].dot(vector_mag)
    B_mag_cart_EASX[i] = vector_mag_cart_EASX

    vector_mag_sphe_EASX = fx.cartToSphere(vector_mag_cart_EASX)
    B_mag_sphe_EASX[i] = vector_mag_sphe_EASX

    B_mag_elev_bin_parallel = fx.binFinder(vector_mag_sphe_EASX[1], bin_dictionary[head]['ELEVATION_lower_bound'], bin_dictionary[head]['ELEVATION_upper_bound'])
    B_mag_elev_bin_antiparallel = fx.binFinder(-vector_mag_sphe_EASX[1], bin_dictionary[head]['ELEVATION_lower_bound'], bin_dictionary[head]['ELEVATION_upper_bound'])
    B_mag_elev_bin[i] = np.array([B_mag_elev_bin_parallel, B_mag_elev_bin_antiparallel])

    B_mag_azim_bin_parallel = fx.binFinder(vector_mag_sphe_EASX[2], bin_dictionary[head]['AZIMUTH_lower_bound'], bin_dictionary[head]['AZIMUTH_upper_bound'])
    B_mag_azim_bin_antiparallel = fx.binFinder((vector_mag_sphe_EASX[2] + 180) % 360, bin_dictionary[head]['AZIMUTH_lower_bound'], bin_dictionary[head]['AZIMUTH_upper_bound'])
    B_mag_azim_bin[i] = np.array([B_mag_azim_bin_parallel, B_mag_azim_bin_antiparallel])

    B_mag_elev_EASX[i] = np.array([bin_dictionary[head]['ELEVATION'][B_mag_elev_bin_parallel], bin_dictionary[head]['ELEVATION'][B_mag_elev_bin_antiparallel]])
    B_mag_azim_EASX[i] = np.array([bin_dictionary[head]['AZIMUTH'][B_mag_azim_bin_parallel], bin_dictionary[head]['AZIMUTH'][B_mag_azim_bin_antiparallel]])

    # calculate EASX elevation and azimuth bins for MAG vectors:
    head = 0
    head, B_eas_EASXz_angle[0][i], B_eas_EASXz_angle[1][i] = fx.headPicker(vector_eas, EASXz_SRF[0], EASXz_SRF[1])
    B_eas_head[i] = head

    vector_eas_cart_EASX = SRFtoEASX[head].dot(vector_eas)
    B_eas_cart_EASX[i] = vector_eas_cart_EASX

    vector_eas_sphe_EASX = fx.cartToSphere(vector_eas_cart_EASX)
    B_eas_sphe_EASX[i] = vector_eas_sphe_EASX

    B_eas_elev_bin_parallel = fx.binFinder(vector_eas_sphe_EASX[1], bin_dictionary[head]['ELEVATION_lower_bound'], bin_dictionary[head]['ELEVATION_upper_bound'])
    B_eas_elev_bin_antiparallel = fx.binFinder(-vector_eas_sphe_EASX[1], bin_dictionary[head]['ELEVATION_lower_bound'], bin_dictionary[head]['ELEVATION_upper_bound'])
    B_eas_elev_bin[i] = np.array([B_eas_elev_bin_parallel, B_eas_elev_bin_antiparallel])

    B_eas_azim_bin_parallel = fx.binFinder(vector_eas_sphe_EASX[2], bin_dictionary[head]['AZIMUTH_lower_bound'], bin_dictionary[head]['AZIMUTH_upper_bound'])
    B_eas_azim_bin_antiparallel = fx.binFinder((vector_eas_sphe_EASX[2] + 180) % 360, bin_dictionary[head]['AZIMUTH_lower_bound'], bin_dictionary[head]['AZIMUTH_upper_bound'])
    B_eas_azim_bin[i] = np.array([B_eas_azim_bin_parallel, B_eas_azim_bin_antiparallel])

    B_eas_elev_EASX[i] = np.array([bin_dictionary[head]['ELEVATION'][B_eas_elev_bin_parallel], bin_dictionary[head]['ELEVATION'][B_eas_elev_bin_antiparallel]])
    B_eas_azim_EASX[i] = np.array([bin_dictionary[head]['AZIMUTH'][B_eas_azim_bin_parallel], bin_dictionary[head]['AZIMUTH'][B_eas_azim_bin_antiparallel]])

    #print('{}, {}'.format(B_mag_elev_bin_parallel,B_mag_elev_bin_antiparallel))
    #print('{}, {}'.format(B_mag_azim_bin_parallel,B_mag_azim_bin_antiparallel))
    #print('{}, {}'.format(B_eas_elev_bin_parallel,B_eas_elev_bin_antiparallel))
    #print('{}, {}'.format(B_eas_azim_bin_parallel,B_eas_azim_bin_antiparallel))

print('\n')

B_eas_elev_bin_parallel, B_eas_elev_bin_antiparallel = B_eas_elev_bin.T # split calculated elevation bin indices (parallel and antiparallel)
B_mag_elev_bin_parallel, B_mag_elev_bin_antiparallel = B_mag_elev_bin.T

B_eas_azim_bin_parallel, B_eas_azim_bin_antiparallel = B_eas_azim_bin.T # split calculated azimuth bin indices (parallel and antiparallel)
B_mag_azim_bin_parallel, B_mag_azim_bin_antiparallel = B_mag_azim_bin.T

B_eas_elev_EASX_parallel, B_eas_elev_EASX_antiparallel = B_eas_elev_EASX.T # split calculated elevation values (parallel and antiparallel)
B_mag_elev_EASX_parallel, B_mag_elev_EASX_antiparallel = B_mag_elev_EASX.T

B_eas_azim_EASX_parallel, B_eas_azim_EASX_antiparallel = B_eas_azim_EASX.T # split calculated azimuth values (parallel and antiparallel)
B_mag_azim_EASX_parallel, B_mag_azim_EASX_antiparallel = B_mag_azim_EASX.T


'''plot configuration'''
geometry = 'spherical' # 'cartesian', 'spherical'
coordinates = 'EASX' # 'SRF','EASX'
print('plotting in {} coordinates with {} geometry'.format(coordinates,geometry))

Nplots = 4 # number of plots to show
sns.set_theme(style='ticks')
fig1, axs = plt.subplots(Nplots)

axs[0].set_title('{}    &    {}'.format(cdf_filename_eas, cdf_filename_mag))


'''coordinates dictionary'''
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
# unit B_radius comparison
ax = axs[0]
ax.plot(time_mag,Bx_mag)
ax.plot(time_eas,Bx_eas)
ax.plot(time_mag,Bx_mag, color='blue')
ax.plot(time_eas,Bx_eas, color='orange')
ax.set_ylabel(r'$B_{r}$'+'\n(unit {})'.format(coordinates))
ax.legend([r"$B_{MAG,r}$", r"$B_{EAS,r}$"])
ax.set_ylim(-0.2,2.2)
tick_spacing = 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.grid()

# unit B_elevation comparison
ax = axs[1]
ax.plot(time_mag,By_mag)
ax.plot(time_eas,By_eas)
ax.plot(time_eas,B_eas_elevation_used_parallel)
#ax.plot(time_eas,B_eas_elevation_used_antiparallel)
ax.plot(time_eas,B_eas_elev_EASX_parallel)
#ax.plot(time_eas,B_eas_elev_EASX_antiparallel)
#ax.plot(time_mag,B_mag_elev_EASX_parallel)
#ax.plot(time_mag,B_mag_elev_EASX_antiparallel)
#ax.plot(time_mag,By_mag, color='blue')
#ax.plot(time_eas,By_eas, color='orange')
ax.set_ylabel(r'$B_{θ}$'+'\n(degrees {})'.format(coordinates))
ax.legend([r"$B_{MAG,θ}$", r"$B_{EAS,θ}$", r"$B_{EAS:Used,θ↑↑}$",r"$B_{EAS:Calc,θ↑↑}$"])#, r"$B_{EAS:Used,θ↑↓}$",r"$B_{EAS:Calc,θ↑↑}$",r"$B_{EAS:Calc,θ↑↓}$"])
ax.set_ylim(-55,55)
tick_spacing = 15 # degrees
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.grid()

# unit B_azimuth comparison
ax = axs[2]
ax.plot(time_mag,Bz_mag)
ax.plot(time_eas,Bz_eas)
ax.plot(time_eas,B_eas_azim_EASX_parallel)
#ax.plot(time_eas,B_eas_azim_EASX_antiparallel)
#ax.plot(time_mag,B_mag_azim_EASX_parallel)
#ax.plot(time_mag,B_mag_azim_EASX_antiparallel)
#ax.plot(time_mag,Bz_mag, color='blue')
#ax.plot(time_eas,Bz_eas, color='orange')
ax.set_ylabel(r'$B_{φ}$'+'\n(degrees {})'.format(coordinates))
ax.legend([r"$B_{MAG,φ}$", r"$B_{EAS,φ}$",r"$B_{EAS:Calc,φ↑↑}$"])#,r"$B_{EAS:Calc,φ↑↓}$"])
ax.set_ylim(-15,375)
tick_spacing = 45 # degrees
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.grid()

# EAS sensor head used
ax = axs[3]
ax.plot(time_eas,eas_used,color='green')
ax.plot(time_eas,B_eas_head,color='purple')
#ax.plot(time_mag,B_mag_head,color='brown')
ax.set_ylabel('Sensor Used')
ax.legend(['EAS Head Used', 'EAS Head Calculated'])#, 'MAG Head Calculated'])
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