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


'''CDF configuration'''
cdf_filename_eas = 'solo_L1_swa-eas-padc_20231105T172733-20231105T174232_V01' + '.cdf' #'solo_L1_swa-eas-padc_20231105T172733-20231105T174232_V01'
cdf_filename_mag = 'solo_L2_mag-srf-normal_20231105_V01' + '.cdf' #'solo_L2_mag-srf-normal_20231105_V01'


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


'''time series processing'''
# raw basic time series
B_eas = cdf_eas['SWA_EAS_MagDataUsed'] # onboard EAS B (SRF, cart)
B_mag = cdf_mag['B_SRF'] # reported MAG B (SRF, cart)

# head used
eas_used = cdf_eas['SWA_EAS_EasUsed'] # the head used for B_eas in the actual data

# EASX elevation bin indices used in various frames and geometries
B_eas_elevation_bin_used_parallel, B_eas_elevation_bin_used_antiparallel = np.array(cdf_eas['SWA_EAS_ElevationUsed']).T

# EASX elevation used in various frames and geometries
B_eas_elevation_used_parallel, B_eas_elevation_used_antiparallel = np.array(cdf_eas['SWA_EAS_ELEVATION']).T # (EASX, sphere)

# series to crop
mag_series_crop_array = [B_mag]
eas_series_crop_array = [B_eas,
                    eas_used,
                    B_eas_elevation_bin_used_parallel,
                    B_eas_elevation_bin_used_antiparallel,
                    B_eas_elevation_used_parallel,
                    B_eas_elevation_used_antiparallel]

# processing basic time series
# if eas_used[0] >  1:
#     time_mag = time_mag[3:]
#     time_eas = time_eas[3:]
#     for i in range(len(mag_series_crop_array)):
#         mag_series_crop_array[i] = mag_series_crop_array[i][3:]
#     for i in range(len(eas_series_crop_array)):
#         eas_series_crop_array[i] = eas_series_crop_array[i][3:]

# mag_series_crop_array, time_mag, eas_series_crop_array, time_eas = fx.cropTimeToOverlap(mag_series_crop_array, time_mag, eas_series_crop_array, time_eas, searchdivisions=5)

t0, tf = fx.getFilenameDatetime_EAS(cdf_filename_eas)
print('final time series from {} to {}'.format(t0, tf))
mag_series_crop_array, time_mag = fx.cropTimeToRef(mag_series_crop_array, time_mag, t0, tf)# + datetime.timedelta(microseconds=87500))
eas_series_crop_array, time_eas = fx.cropTimeToRef(eas_series_crop_array, time_eas, t0, tf)# + datetime.timedelta(microseconds=87500))

B_mag = mag_series_crop_array[0]
B_eas = eas_series_crop_array[0]
eas_used = eas_series_crop_array[1]
B_eas_elevation_bin_used_parallel = eas_series_crop_array[2]
B_eas_elevation_bin_used_antiparallel = eas_series_crop_array[3]
B_eas_elevation_used_parallel = eas_series_crop_array[4]
B_eas_elevation_used_antiparallel = eas_series_crop_array[5]

len_B_eas = len(B_eas)
len_B_mag = len(B_mag)
print('\nB_eas vector count = {}\nB_mag vector count = {}'.format(len_B_eas, len_B_mag))

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

# head calculated
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

# calculated EASX elevation bin indices (parallel and antiparallel)
B_eas_elev_bin = np.ndarray((len_B_eas,2))
B_mag_elev_bin = np.ndarray((len_B_mag,2))

# calculated EASX elevation (parallel and antiparallel)
B_eas_elev_EASX = np.ndarray((len_B_eas,2))
B_mag_elev_EASX = np.ndarray((len_B_mag,2))

B_eas_elev_EASX_upper = np.ndarray((len_B_eas,2))
B_mag_elev_EASX_upper = np.ndarray((len_B_mag,2))

B_eas_elev_EASX_lower = np.ndarray((len_B_eas,2))
B_mag_elev_EASX_lower = np.ndarray((len_B_mag,2))

# calculated EAS azimuth bin index (parallel and antiparallel)
B_eas_azim_bin = np.ndarray((len_B_eas,2))
B_mag_azim_bin = np.ndarray((len_B_eas,2))

# calculated EASX azimuth (parallel and antiparallel)
B_eas_azim_EASX = np.ndarray((len_B_eas,2))
B_mag_azim_EASX = np.ndarray((len_B_mag,2))

B_eas_azim_EASX_upper = np.ndarray((len_B_eas,2))
B_mag_azim_EASX_upper = np.ndarray((len_B_mag,2))

B_eas_azim_EASX_lower = np.ndarray((len_B_eas,2))
B_mag_azim_EASX_lower = np.ndarray((len_B_mag,2))

print('\n')
for i in range(0,len_B_eas):
        print('\rtime steps processed = {}/{}'.format(i+1,len_B_eas), end='')

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
        #print('\n{}'.format(eas_used[i]))
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

        # calculate EASX elevation and azimuth bins for MAG vectors:
        head_mag, B_mag_EASXz_angle[0][i], B_mag_EASXz_angle[1][i] = fx.headPicker(vector_mag, EASXz_SRF[0], EASXz_SRF[1])
        B_mag_head[i] = head_mag
        
        vector_mag_cart_EASX = SRFtoEASX[head_mag].dot(vector_mag)
        B_mag_cart_EASX[i] = vector_mag_cart_EASX

        vector_mag_sphe_EASX = fx.cartToSphere(vector_mag_cart_EASX)
        B_mag_sphe_EASX[i] = vector_mag_sphe_EASX

        B_mag_elev_bin[i], B_mag_azim_bin[i], B_mag_elev_EASX[i], B_mag_azim_EASX[i] = fx.binFinder(vector_mag_sphe_EASX, head_mag, old_bin_dictionary)#old_bin_dictionary)

        # calculate EASX elevation and azimuth bins for EAS vectors:
        head_eas, B_eas_EASXz_angle[0][i], B_eas_EASXz_angle[1][i] = fx.headPicker(vector_eas, EASXz_SRF[0], EASXz_SRF[1])
        B_eas_head[i] = head_eas

        vector_eas_cart_EASX = SRFtoEASX[eas_used[i]].dot(vector_eas)
        B_eas_cart_EASX[i] = vector_eas_cart_EASX

        vector_eas_sphe_EASX = fx.cartToSphere(vector_eas_cart_EASX)
        B_eas_sphe_EASX[i] = vector_eas_sphe_EASX

        #B_eas_elev_bin[i], B_eas_azim_bin[i], B_eas_elev_EASX[i], B_eas_azim_EASX[i] = fx.binFinder(vector_eas_sphe_EASX, head_eas, bin_dictionary)#old_bin_dictionary)#B_eas_spherical_EASX[i], eas_used[i], bin_dictionary)#fx.binFinder(vector_eas_sphe_EASX, head_eas, bin_dictionary)#old_bin_dictionary)
        B_eas_elev_bin[i], B_eas_azim_bin[i], B_eas_elev_EASX[i], B_eas_azim_EASX[i] = fx.binFinder(B_eas_spherical_EASX[i], eas_used[i], old_bin_dictionary)#old_bin_dictionary)#B_eas_spherical_EASX[i], eas_used[i], bin_dictionary)#fx.binFinder(vector_eas_sphe_EASX, head_eas, bin_dictionary)#old_bin_dictionary)
        temp1, temp2, B_eas_elev_EASX_upper[i], B_eas_azim_EASX_upper[i] = fx.binFinder_UpperBound(B_eas_spherical_EASX[i], eas_used[i], old_bin_dictionary)#old_bin_dictionary)
        temp1, temp2, B_eas_elev_EASX_lower[i], B_eas_azim_EASX_lower[i] = fx.binFinder_LowerBound(B_eas_spherical_EASX[i], eas_used[i], old_bin_dictionary)#old_bin_dictionary)

        # if (B_eas_elev_EASX[i][0] == -1.66) or (B_eas_elev_EASX[i][0] == -1.93):
        #     print('\ngoteem')

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
B_eas_elev_EASX_upper_parallel, B_eas_elev_EASX_upper_antiparallel = B_eas_elev_EASX_upper.T
B_eas_elev_EASX_lower_parallel, B_eas_elev_EASX_lower_antiparallel = B_eas_elev_EASX_lower.T
B_mag_elev_EASX_parallel, B_mag_elev_EASX_antiparallel = B_mag_elev_EASX.T

B_eas_azim_EASX_parallel, B_eas_azim_EASX_antiparallel = B_eas_azim_EASX.T # split calculated azimuth values (parallel and antiparallel)
B_mag_azim_EASX_parallel, B_mag_azim_EASX_antiparallel = B_mag_azim_EASX.T


# sanity check:
san_i = 100 # sanity check index
sanity_dictionary = {'B,cart,SRF':[B_eas[san_i], B_eas[san_i]],
                'head':[eas_used[san_i], B_eas_head[san_i]],
                'B,cart,EASX':[B_eas_EASX[san_i],B_eas_cart_EASX[san_i]],
                'B,sphe,EASX':[B_eas_spherical_EASX[san_i], B_eas_sphe_EASX[san_i]],
                'Elevation Parallel Bin Index':[B_eas_elevation_bin_used_parallel[san_i], B_eas_elev_bin_parallel[san_i]],
                'Elevation Parallel Bin Value':[B_eas_elevation_used_parallel[san_i], B_eas_elev_EASX_parallel[san_i]]}

for san_i in range(3):
        san_i = int(san_i*len_B_eas/3)
        sanity_dictionary = {'B,cart,SRF':[B_eas[san_i], B_eas[san_i]],
                'head':[eas_used[san_i], B_eas_head[san_i]],
                'B,cart,EASX':[B_eas_EASX[san_i],B_eas_cart_EASX[san_i]],
                'B,sphe,EASX':[B_eas_spherical_EASX[san_i], B_eas_sphe_EASX[san_i]],
                'Elevation Parallel Bin Index':[B_eas_elevation_bin_used_parallel[san_i], B_eas_elev_bin_parallel[san_i]],
                'Elevation Parallel Bin Value':[B_eas_elevation_used_parallel[san_i], B_eas_elev_EASX_parallel[san_i]]}
        print('\n')
        print('time = {}'.format(time_eas[san_i]))
        for key in sanity_dictionary:
                print('{}[{}]: SOAr\'s = {}, mine = {}'.format(key, san_i, sanity_dictionary[key][0], sanity_dictionary[key][1]))
        print('\n')


'''plot configuration'''
geometry = 'spherical' # 'cartesian', 'spherical'
coordinates = 'EASX' # 'SRF','EASX'
print('plotting in {} coordinates with {} geometry'.format(coordinates,geometry))

Nplots = 3 # number of plots to show
sns.set_theme(style='ticks')
fig1, axs = plt.subplots(Nplots)

axs[0].set_title('{}    &    {}'.format(cdf_filename_eas, cdf_filename_mag))


'''coordinates dictionary'''
coordinates_dictionary = {'SRF':{'cartesian':(B_mag,B_eas),'spherical':(B_mag_spherical_SRF,B_eas_spherical_SRF)}, 
                        'EASX':{'cartesian':(B_mag_EASX,B_eas_EASX),'spherical':(B_mag_spherical_EASX,B_eas_spherical_EASX)}}
Bx_mag, By_mag, Bz_mag = np.array(coordinates_dictionary[coordinates][geometry][0]).T
Bx_eas, By_eas, Bz_eas = np.array(coordinates_dictionary[coordinates][geometry][1]).T

# temp:
Bx_mag_calc, By_mag_calc, Bz_mag_calc = B_mag_sphe_EASX.T
Bx_eas_calc, By_eas_calc, Bz_eas_calc = B_eas_sphe_EASX.T


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
# # unit B_radius comparison
# ax = axs[0]
# ax.plot(time_mag,Bx_mag)
# ax.plot(time_eas,Bx_eas)
# #ax.plot(time_mag,Bx_mag_calc, color='blue')
# #ax.plot(time_eas,Bx_eas_calc, color='orange')
# ax.set_ylabel('B Amplitude'+'\n(unit {})'.format(coordinates))
# ax.legend([r"$B_{MAG}$", r"$B_{EAS}$"])#, r"$B_{MAG:Calc}$", r"$B_{EAS:Calc}$"])
# ax.set_ylim(-0.2,2.2)
# tick_spacing = 1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# ax.grid()

# # unit B_elevation comparison
# ax = axs[1]
# ax.plot(time_mag,By_mag)
# ax.plot(time_eas,By_eas)
# ax.plot(time_mag,By_mag_calc, color='blue')
# ax.plot(time_eas,By_eas_calc, color='orange')
# ax.set_ylabel(r'$B_{θ}$'+'\n(degrees {})'.format(coordinates))
# ax.legend([r"$B_{MAG:Used,θ↑↑}$", r"$B_{EAS:Used,θ↑↑}$", r"$B_{MAG:Calc,θ↑↑}$", r"$B_{EAS:Calc,θ↑↑}$"])
# ax.set_ylim(-55,55)
# tick_spacing = 15 # degrees
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# ax.grid()

# # unit B_azimuth comparison
# ax = axs[2]
# ax.plot(time_mag,Bz_mag)
# ax.plot(time_eas,Bz_eas)
# ax.plot(time_mag,Bz_mag_calc, color='blue')
# ax.plot(time_eas,Bz_eas_calc, color='orange')
# ax.set_ylabel(r'$B_{φ}$'+'\n(degrees {})'.format(coordinates))
# ax.legend([r"$B_{MAG:Used,φ↑↑}$", r"$B_{EAS:Used,φ↑↑}$", r"$B_{MAG:Calc,φ↑↑}$", r"$B_{EAS:Calc,φ↑↑}$"])
# ax.set_ylim(-15,375)
# tick_spacing = 45 # degrees
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# ax.grid()

# # # unit B_elevation comparison
# ax = axs[1]
# # ax.plot(time_mag,By_mag)
# #ax.plot(time_eas,By_eas)
# ax.plot(time_eas,B_eas_elevation_used_parallel)
# ax.plot(time_eas,B_eas_elev_EASX_parallel, color='green')
# ax.plot(time_eas,B_eas_elevation_used_antiparallel)
# ax.plot(time_eas,B_eas_elev_EASX_antiparallel, color='purple')
# #ax.plot(time_eas,B_eas_elev_EASX_upper_parallel, color='gray')
# #ax.plot(time_eas,B_eas_elev_EASX_lower_parallel, color='gray')
# #ax.plot(time_mag,B_mag_elev_EASX_parallel)
# #ax.plot(time_mag,B_mag_elev_EASX_antiparallel)
# #ax.plot(time_mag,By_mag, color='blue')
# #ax.plot(time_eas,By_eas, color='orange')
# ax.set_ylabel('B Elevation'+'\n(degrees {})'.format(coordinates))
# ax.legend([r"$B_{EAS:Used,↑↑}$", r"$B_{EAS:Calc,↑↑}$", r"$B_{EAS:Used,↑↓}$", r"$B_{EAS:Calc,↑↓}$"])#r"$B_{MAG,θ}$", r"$B_{EAS,θ}$", r"$B_{EAS:Used,θ↑↑}$", r"$B_{EAS:Calc,θ↑↑}$", r"$B_{EAS:Used,θ↑↓}$", r"$B_{EAS:Calc,θ↑↓}$"])#, r"$B_{EAS:Used,θ↑↓}$",r"$B_{EAS:Calc,θ↑↑}$",r"$B_{EAS:Calc,θ↑↓}$"])
# ax.set_ylim(-55,55)
# tick_spacing = 15 # degrees
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# ax.grid()

# # unit B_azimuth comparison
# ax = axs[2]
# # ax.plot(time_mag,Bz_mag)
# ax.plot(time_eas,Bz_eas)
# ax.plot(time_eas,B_eas_azim_EASX_parallel, color='green')
# ax.plot(time_eas,B_eas_azim_EASX_antiparallel, color='purple')
# #ax.plot(time_mag,B_mag_azim_EASX_parallel)
# #ax.plot(time_mag,B_mag_azim_EASX_antiparallel)
# #ax.plot(time_mag,Bz_mag, color='blue')
# #ax.plot(time_eas,Bz_eas, color='orange')
# ax.set_ylabel('B Azimuth'+'\n(degrees {})'.format(coordinates))
# ax.legend([r"$B_{EAS}$", r"$B_{EAS:Calc,↑↑}$", r"$B_{EAS:Calc,↑↓}$"])#, r"$B_{EAS:Calc,φ↑↓}$"])#r"$B_{MAG,φ}$", r"$B_{EAS,φ}$",r"$B_{EAS:Calc,φ↑↑}$",r"$B_{EAS:Calc,φ↑↓}$"])
# ax.set_ylim(-15,375)
# tick_spacing = 45 # degrees
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# ax.grid()

# # unit B_elevation bin comparison
ax = axs[0]
# ax.plot(time_mag,By_mag)
#ax.plot(time_eas,By_eas)
ax.plot(time_eas,B_eas_elevation_bin_used_parallel)
ax.plot(time_eas,B_eas_elev_bin_parallel, color='green')
ax.plot(time_eas,B_eas_elevation_bin_used_antiparallel)
ax.plot(time_eas,B_eas_elev_bin_antiparallel, color='purple')
#ax.plot(time_eas,B_eas_elev_EASX_upper_parallel, color='gray')
#ax.plot(time_eas,B_eas_elev_EASX_lower_parallel, color='gray')
#ax.plot(time_mag,B_mag_elev_EASX_parallel)
#ax.plot(time_mag,B_mag_elev_EASX_antiparallel)
#ax.plot(time_mag,By_mag, color='blue')
#ax.plot(time_eas,By_eas, color='orange')
ax.set_ylabel('B Elevation\nBin Index')
ax.legend([r"$B_{EAS:Used,↑↑}$", r"$B_{EAS:Calc,↑↑}$", r"$B_{EAS:Used,↑↓}$", r"$B_{EAS:Calc,↑↓}$"])#r"$B_{MAG,θ}$", r"$B_{EAS,θ}$", r"$B_{EAS:Used,θ↑↑}$", r"$B_{EAS:Calc,θ↑↑}$", r"$B_{EAS:Used,θ↑↓}$", r"$B_{EAS:Calc,θ↑↓}$"])#, r"$B_{EAS:Used,θ↑↓}$",r"$B_{EAS:Calc,θ↑↑}$",r"$B_{EAS:Calc,θ↑↓}$"])
ax.set_ylim(-1,16)
tick_spacing = 1 # degrees
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.grid()

# unit B_azimuth bin comparison
ax = axs[1]
# ax.plot(time_mag,Bz_mag)
#ax.plot(time_eas,Bz_eas)
ax.plot(time_eas,B_eas_azim_bin_parallel, color='green')
ax.plot(time_eas,B_eas_azim_bin_antiparallel, color='purple')
#ax.plot(time_mag,B_mag_azim_EASX_parallel)
#ax.plot(time_mag,B_mag_azim_EASX_antiparallel)
#ax.plot(time_mag,Bz_mag, color='blue')
#ax.plot(time_eas,Bz_eas, color='orange')
ax.set_ylabel('B Azimuth\nBin Index')
ax.legend([r"$B_{EAS:Calc,↑↑}$", r"$B_{EAS:Calc,↑↓}$"])#, r"$B_{EAS:Calc,φ↑↓}$"])#r"$B_{MAG,φ}$", r"$B_{EAS,φ}$",r"$B_{EAS:Calc,φ↑↑}$",r"$B_{EAS:Calc,φ↑↓}$"])
ax.set_ylim(-2,33)
tick_spacing = 2 # bins
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.grid()

# EAS sensor head used
ax = axs[Nplots-1]
ax.plot(time_eas,eas_used,color='green')
#ax.plot(time_eas,B_eas_head,color='purple')
#ax.plot(time_mag,B_mag_head,color='brown')
ax.set_ylabel('Sensor Used')
ax.legend(['EAS Head Used'])#, 'EAS Head Calculated'])#, 'MAG Head Calculated'])
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