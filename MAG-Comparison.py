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
    B_mag_EASX[i] = np.array(vector_mag_magnitude_EASX)
    B_eas_EASX[i] = np.array(vector_eas_magnitude_EASX)

Bx_mag, By_mag, Bz_mag = np.array(B_mag).T
Bx_eas, By_eas, Bz_eas = np.array(B_eas).T

Bx_mag_EASX, By_mag_EASX, Bz_mag_EASX = np.array(B_mag_EASX).T
Bx_eas_EASX, By_eas_EASX, Bz_eas_EASX = np.array(B_eas_EASX).T

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

sr = 8

delayBx, lagsBx, corrBx = delayFinder(np.pad(Bx_mag,(0,0)),Bx_eas[:],sr)
delayBy, lagsBy, corrBy = delayFinder(np.pad(By_mag,(0,0)),By_eas[:],sr)
delayBz, lagsBz, corrBz = delayFinder(np.pad(Bz_mag,(0,0)),Bz_eas[:],sr)

#print('B delays (s) = ({},{},{})'.format(delayBx,delayBy,delayBz))
#print('B delays (s) = ({},{},{})'.format(delayFinder2(Bx_mag,Bx_eas,sr), delayFinder2(By_mag,Bz_eas,sr), delayFinder2(Bz_mag,Bz_eas,sr)))

for i in range(len(corrBx)):
    corrBx[i] = np.linalg.norm(corrBx[i])
    corrBy[i] = np.linalg.norm(corrBy[i])
    corrBz[i] = np.linalg.norm(corrBz[i])

coordinates = 'SRF' # 'SRF','EASX'
if coordinates == 'SRF':
    Bx_mag, By_mag, Bz_mag = Bx_mag, By_mag, Bz_mag
    Bx_eas, By_eas, Bz_eas = Bx_eas, By_eas, Bz_eas
elif coordinates == 'EASX':
    Bx_mag, By_mag, Bz_mag = Bx_mag_EASX, By_mag_EASX, Bz_mag_EASX
    Bx_eas, By_eas, Bz_eas = Bx_eas_EASX, By_eas_EASX, Bz_eas_EASX

Nplots = 4
sns.set_theme(style='ticks')
fig1, axs = plt.subplots(Nplots)

axs[0].set_title('{}    &    {}'.format(cdf_filename_eas, cdf_filename_mag))

axs[0].plot(time_mag,Bx_mag)
axs[0].plot(time_eas,Bx_eas)
axs[0].set_ylabel(r'$B_{x}$'+'\n(unit {})'.format(coordinates))
axs[0].legend([r"$B_{MAG,x}$", r"$B_{EAS,x}$"])
axs[0].set_ylim(-1.2,1.2)
axs[0].grid()

axs[1].plot(time_mag,By_mag)
axs[1].plot(time_eas,By_eas)
axs[1].set_ylabel(r'$B_{y}$'+'\n(unit {})'.format(coordinates))
axs[1].legend([r"$B_{MAG,y}$", r"$B_{EAS,y}$"])
axs[1].set_ylim(-1.2,1.2)
axs[1].grid()

axs[2].plot(time_mag,Bz_mag)
axs[2].plot(time_eas,Bz_eas)
axs[2].set_ylabel(r'$B_{z}$'+'\n(unit {})'.format(coordinates))
axs[2].legend([r"$B_{MAG,z}$", r"$B_{EAS,z}$"])
axs[2].set_ylim(-1.2,1.2)
axs[2].grid()

axs[3].plot(time_eas,eas_used,color='green')
axs[3].set_ylabel('Sensor Used')
axs[3].set_ylim(-0.2,1.2)
axs[3].set_yticks([0,1],['EAS1','EAS2'])
axs[3].grid()

'''
axs[3].plot(time_mag,B_angle, color='green')
#axs[3].plot(time_eas[:-1],B_angle, color='red') # in case you want to compare the two same series! (one is  currently one point shorter, for some reason)
axs[3].set_ylabel('angle between\nB vectors (°)')
#tick_spacing = 2 # degrees
#axs[3].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
axs[3].legend([r'$B_{MAG}-B_{EAS}$ Angle'])
axs[3].grid()

axs[Nplots-1].set_xlabel('Date & Time\n(dd hh:mm)')#axs[Nplots-1].set_xlabel('Date & Time (dd hh:mm)')
'''
'''
axs[0].plot(time_mag,Bx_mag)
axs[0].plot(time_eas,Bx_eas)
axs[0].plot(time_mag,By_mag)
axs[0].plot(time_eas,By_eas)
axs[0].plot(time_mag,Bz_mag)
axs[0].plot(time_eas,Bz_eas)
axs[0].set_ylabel('B\n(normalised)')
axs[0].legend(["Bx MAG", "Bx EAS","By MAG", "By EAS","Bz MAG", "Bz EAS"])
axs[0].set_ylim(-1.2,1.2)
axs[0].grid()
'''
'''
axs[3].plot(time_mag,B_mag_magnitude)
axs[3].plot(time_eas,B_eas_magnitude)
axs[3].set_ylabel('|B|\n(nT)')
axs[3].legend(["|B| MAG", "|B| EAS"])
axs[3].set_ylim(-1.2,1.2)
axs[3].grid()
'''

'''
#axs[Nplots-1].plot(lagsBx,corrBx)
#axs[Nplots-1].plot(lagsBy,corrBy)
#axs[Nplots-1].plot(lagsBz,corrBz)
#axs[Nplots-1].set_ylabel('correlation\n(dimensionless)')
#axs[Nplots-1].set_xlabel('possible time delays (seconds)')
#axs[Nplots-1].set_yticklabels([''])
#axs[Nplots-1].legend(["Bx Correlation", "By Correlation", "Bz Correlation"])
#axs[Nplots-1].grid()

plt.show(block=True) # set to block=False to automatically close figure
'''

plt.show()