import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
os.environ["CDF_LIB"] = "C:\\Program Files\\CDF_Distribution"
from spacepy import pycdf
import datetime

cdf_filename_eas = 'solo_L1_swa-eas-padc_20231105T172733-20231105T174232_V01.cdf'
cdf_eas = pycdf.CDF('data\\' + cdf_filename_eas)
#print(cdf_eas)
time_eas = np.array(cdf_eas['EPOCH'])
#print(time_eas)
t0 = time_eas[0] # start of timeframe
tf = time_eas[-1] # end of timeframe
print('EAS time series from {} to {}'.format(t0, tf))

cdf_filename_mag = 'solo_L2_mag-srf-normal_20231105_V01.cdf'#'solo_L2_mag-srf-normal_20230831_V01.cdf'
cdf_mag = pycdf.CDF('data\\' + cdf_filename_mag)
#print(cdf_mag)
time_mag = cdf_mag['EPOCH']#[500000:])

t0_eas = time_eas[0] # start of B EAS timeframe
t0_mag = time_mag[0] # start of B MAG timeframe
tf_eas = time_eas[-1] # end of B EAS timeframe
tf_mag = time_mag[-1] # end of B MAG timeframe


def cropTime_indexFinder(timeArray,timeReference,searchdivisions=5):
    # this function finds a time index given a time, efficiently.

    # timeArray is the time array where you're finding an index.
    # timeReference is the reference time whose index you're trying to find.
    # searchdivisions is the number of times timeArray should be divided to narrow the search space.
    length = len(timeArray)
    time_index = 0
    for i in range(1,searchdivisions+1):
        time_index += int(length/2**i)*(timeArray[time_index+int(length/2**i)] < t0) # add smaller and smaller slices to converge on the right time index for t0
    while timeArray[time_index] < timeReference:
        time_index += 1
        print('\rtime index = {}/{}'.format(time_index,length), end='')
        continue
    print('\n')

    return time_index


def cropTimeToOverlap(seriesA,timeA,seriesB,timeB,searchdivisions=5):
    # crops two time series to the period where they overlap in time

    if len(timeA) != len(seriesA) or len(timeB) != len(seriesB):
        raise Exception('all time and series axes must be the same size!')

    t0A = timeA[0] # timeA start
    t0B = timeB[0] # timeB start
    tfA = timeA[-1] # timeA finish
    tfB = timeB[-1] # timeB finish

    if t0A > tfB or t0B > tfA:
        raise Exception('these time series do not overlap!')

    print('\n')
    print('t0A = {}'.format(t0A))
    print('t0B = {}'.format(t0B))

    if t0A > t0B: # if t0A is later than t0B, start at t0A, therefore series B starts early.
        print('overlap starts at {}'.format(t0A))
        time_index_t0 = cropTime_indexFinder(timeB, t0A, searchdivisions)
        timeB = timeB[time_index_t0:]
        seriesB = seriesB[time_index_t0:]
        
    else: # vice versa.
        print('overlap starts at {}'.format(t0B))
        time_index_t0 = cropTime_indexFinder(timeA, t0B, searchdivisions)
        timeA = timeA[time_index_t0:]
        seriesA = seriesA[time_index_t0:]
        

    print('\n')
    print('tfA = {}'.format(tfA))
    print('tfB = {}'.format(tfB))

    if tfA > tfB: # if tfA is later than tfB, finish at tfB, therefore series A finishes late.
        print('overlap finishes at {}'.format(t0B))
        time_index_t0 = cropTime_indexFinder(timeA, tfB, searchdivisions)
        timeA = timeA[:time_index_t0]
        seriesA = seriesA[:time_index_t0]
        
    else: # vice versa.
        print('overlap finishes at {}'.format(t0A))
        time_index_t0 = cropTime_indexFinder(timeB, tfA, searchdivisions)
        timeB = timeB[:time_index_t0]
        seriesB = seriesB[:time_index_t0]
        

    return seriesA, timeA, seriesB, timeB


def cartToSphere(vector):
    x, y, z, = vector[0], vector[1], vector[2]
    R = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2+z**2),z) * 180/np.pi
    phi = np.arctan2(y,x) * 180/np.pi
    return np.array([R,theta,phi])


def delayFinder(f,g,sr): 
    # f and g are the functions to cross-correlate
    # sr is the samplerate
    # positive delays when f is ahead

    corr = sp.signal.correlate(f,g,mode='full') # the correlation distribution
    lags = sp.signal.correlation_lags(f.size,g.size,mode='full')/sr # range of possible delays ('lags'), scaled by samplerate
    delay = lags[np.argmax(corr)]

    return delay, lags, corr

#def delayFinder2(data_1,data_2,sr):
#    correlation = np.correlate(data_1, data_2, mode='same')
#    delay = ( np.argmax(correlation) - int(len(correlation)/2) ) / sr
#    return delay

#print('B delays (s) = ({},{},{})'.format(delayBx,delayBy,delayBz))
#print('B delays (s) = ({},{},{})'.format(delayFinder2(Bx_mag,Bx_eas,sr), delayFinder2(By_mag,Bz_eas,sr), delayFinder2(Bz_mag,Bz_eas,sr)))


r2o2 = np.sqrt(2)/2 # root 2 over 2
SRFtoEAS1 = np.array([[0,0,-1],[-r2o2,r2o2,0],[r2o2,r2o2,0]])
SRFtoEAS2 = np.array([[0,0,1],[-r2o2,-r2o2,0],[r2o2,-r2o2,0]])
SRFtoEASX = (SRFtoEAS1,SRFtoEAS2) # the transform matrices for SRF to the respective EAS head coordinates. First is EAS1, second is EAS2.

B_eas = cdf_eas['SWA_EAS_MagDataUsed'] # onboard EAS B (SRF)
B_mag = cdf_mag['B_SRF'] # reported MAG B (SRF)

B_mag, time_mag, B_eas, time_eas = cropTimeToOverlap(B_mag, time_mag, B_eas, time_eas, searchdivisions=5)

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

B_eas_elevation_used_EAS1, B_eas_elevation_used_EAS2 = np.array(cdf_eas['SWA_EAS_ELEVATION']).T

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
    B_mag_spherical_SRF[i] = cartToSphere(vector_mag)
    B_mag_spherical_EASX[i] = cartToSphere(vector_mag_magnitude_EASX)
    B_eas_spherical_SRF[i] = cartToSphere(vector_eas)
    B_eas_spherical_EASX[i] = cartToSphere(vector_eas_magnitude_EASX)

geometry = 'cartesian' # 'cartesian', 'spherical'
coordinates = 'EASX' # 'SRF','EASX'
coordinates_dictionary = {'SRF':{'cartesian':(B_mag,B_eas),'spherical':(B_mag_spherical_SRF,B_eas_spherical_SRF)}, 
                        'EASX':{'cartesian':(B_mag_EASX,B_eas_EASX),'spherical':(B_mag_spherical_EASX,B_eas_spherical_EASX)}}
Bx_mag, By_mag, Bz_mag = np.array(coordinates_dictionary[coordinates][geometry][0]).T
Bx_eas, By_eas, Bz_eas = np.array(coordinates_dictionary[coordinates][geometry][1]).T

sr = 8
delayBx, lagsBx, corrBx = delayFinder(np.pad(Bx_mag,(0,0)),Bx_eas[:],sr)
delayBy, lagsBy, corrBy = delayFinder(np.pad(By_mag,(0,0)),By_eas[:],sr)
delayBz, lagsBz, corrBz = delayFinder(np.pad(Bz_mag,(0,0)),Bz_eas[:],sr)

for i in range(len(corrBx)):
    corrBx[i] = np.linalg.norm(corrBx[i])
    corrBy[i] = np.linalg.norm(corrBy[i])
    corrBz[i] = np.linalg.norm(corrBz[i])

Nplots = 4 # number of plots to show
sns.set_theme(style='ticks')
fig1, axs = plt.subplots(Nplots)

axs[0].set_title('{}    &    {}'.format(cdf_filename_eas, cdf_filename_mag))

'''
# unit BR comparison
ax = axs[0]
ax.plot(time_mag,Bx_mag)
ax.plot(time_eas[:-1],Bx_eas[:-1])
ax.set_ylabel(r'$B_{r}$'+'\n(unit {})'.format(coordinates))
ax.legend([r"$B_{MAG,r}$", r"$B_{EAS,r}$"])
ax.set_ylim(-1.2,1.2)
ax.grid()

# unit Btheta comparison
ax = axs[1]
ax.plot(time_mag,By_mag)
ax.plot(time_eas[:-1],By_eas[:-1])
ax.plot(time_eas[:-1],B_eas_elevation_used_EAS1[:-1])
ax.plot(time_eas[:-1],B_eas_elevation_used_EAS2[:-1])
ax.set_ylabel(r'$B_{θ}$'+'\n(degrees {})'.format(coordinates))
ax.legend([r"$B_{MAG,θ}$", r"$B_{EAS,θ}$", r"$B_{EAS1_Used,θ}$", r"$B_{EAS2_Used,θ}$"])
#ax.set_ylim(-50,50)
ax.grid()

# unit Bphi comparison
ax = axs[2]
ax.plot(time_mag,Bz_mag)
ax.plot(time_eas[:-1],Bz_eas[:-1])
ax.set_ylabel(r'$B_{φ}$'+'\n(degrees {})'.format(coordinates))
ax.legend([r"$B_{MAG,φ}$", r"$B_{EAS,φ}$"])
ax.set_ylim(-180,180)
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