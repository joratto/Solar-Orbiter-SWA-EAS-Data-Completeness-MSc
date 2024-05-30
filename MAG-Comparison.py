import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
os.environ["CDF_LIB"] = "C:\\Program Files\\CDF_Distribution"
from spacepy import pycdf
import datetime

cdf_filename_eas = 'solo_L1_swa-eas-padc_20230530T172732-20230530T173231_V01.cdf'#solo_L1_swa-eas-padc_20230831T172734-20230831T173233_V01.cdf'
cdf_eas = pycdf.CDF('data\\' + cdf_filename_eas)
#print(cdf_eas)
time_eas = np.array(cdf_eas['EPOCH'])
#print(time_eas)
t0 = time_eas[0] # start of timeframe
tf = time_eas[-1] # end of timeframe
print('EAS time series from {} to {}'.format(t0, tf))

cdf_filename_mag = 'solo_L2_mag-srf-normal_20230530_V01.cdf'#'solo_L2_mag-srf-normal_20230831_V01.cdf'
cdf_mag = pycdf.CDF('data\\' + cdf_filename_mag)
#print(cdf_mag)
time_mag = cdf_mag['EPOCH']#[500000:])

def cropTimeToRef(seriesaxis,timeaxis,reftimeaxis,searchdivisions=5):
    # searchdivisions defines the number of times to split timeaxis when searching for t0
    # reftimeaxis is the reference time axis being cropped to
    # timeaxis is the time axis being cropped
    # seriesaxis is the corresponding series being cropped the same as timeaxis

    t0 = reftimeaxis[0] # start of reference timeframe
    tf = reftimeaxis[-1] # end of reference timeframe

    length = len(timeaxis)

    if length != len(seriesaxis):
        raise Exception('time and series axes must be the same size!')
    if len(reftimeaxis) > length:
        raise Exception('reference timeframe must be shorter than the timeframe to be cropped!')
    
    print('\n')
    time_index_t0 = 0
    for i in range(1,searchdivisions+1):
        time_index_t0 += int(length/2**i)*(timeaxis[time_index_t0+int(length/2**i)] < t0) # add smaller and smaller slices to converge on the right time_mag index for t0
    while timeaxis[time_index_t0] < t0:
        if time_index_t0 >= length:
            raise Exception('reference timeframe not contained within the timeframe to be cropped!')
        time_index_t0 += 1
        print('\rt0 index = {}/{}'.format(time_index_t0,length), end='')
        continue
    print('\n')
    time_index_tf = time_index_t0
    while timeaxis[time_index_tf] < tf:
        if time_index_tf >= length:
            break # because the time axis to be cropped ends too early
        time_index_tf += 1
        print('\rtf index = {}/{}'.format(time_index_tf,length), end='')
        continue
    print('\n')

    timeaxisCropped = timeaxis[time_index_t0:time_index_tf]
    seriesaxisCropped = seriesaxis[time_index_t0:time_index_tf]

    return seriesaxisCropped, timeaxisCropped

B_eas = cdf_eas['SWA_EAS_MagDataUsed'] # onboard EAS B
B_mag = cdf_mag['B_SRF'] # reported MAG B
B_mag, time_mag = cropTimeToRef(B_mag,time_mag,time_eas)

B_eas_magnitude = np.ndarray((len(B_eas)))
B_mag_magnitude = np.ndarray((len(B_mag)))

B_angle = np.ndarray((len(B_mag)))

for i in range(len(B_mag)):
    # calculate vectors and magnitudes:
    vector_mag = B_mag[i]
    vector_mag_magnitude = np.linalg.norm(vector_mag)
    B_mag_magnitude[i] = vector_mag_magnitude

    vector_eas = B_eas[i]
    vector_eas_magnitude = np.linalg.norm(vector_eas)
    B_eas_magnitude[i] = vector_eas_magnitude

    # normalise B_mag:
    B_mag[i] = vector_mag/vector_mag_magnitude

    # calculate angles between MAG vectors and EAS vectors:
    B_angle[i] = np.arccos(np.dot(vector_mag,vector_eas))*180/np.pi

Bx_mag, By_mag, Bz_mag = np.array(B_mag).T
Bx_eas, By_eas, Bz_eas = np.array(B_eas).T

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

print('B delays (s) = ({},{},{})'.format(delayBx,delayBy,delayBz))
#print('B delays (s) = ({},{},{})'.format(delayFinder2(Bx_mag,Bx_eas,sr), delayFinder2(By_mag,Bz_eas,sr), delayFinder2(Bz_mag,Bz_eas,sr)))

for i in range(len(corrBx)):
    corrBx[i] = np.linalg.norm(corrBx[i])
    corrBy[i] = np.linalg.norm(corrBy[i])
    corrBz[i] = np.linalg.norm(corrBz[i])

Nplots = 4
sns.set_theme(style='ticks')
fig, axs = plt.subplots(Nplots)

axs[0].set_title('{}    &    {}'.format(cdf_filename_eas, cdf_filename_mag))

axs[0].plot(time_mag,Bx_mag)
axs[0].plot(time_eas,Bx_eas)
axs[0].set_ylabel(r'$B_{x}$'+'\n(normalised)')
axs[0].legend([r"$B_{MAG,x}$", r"$B_{EAS,x}$"])
axs[0].set_ylim(-1.2,1.2)
axs[0].grid()

axs[1].plot(time_mag,By_mag)
axs[1].plot(time_eas,By_eas)
axs[1].set_ylabel(r'$B_{y}$'+'\n(normalised)')
axs[1].legend([r"$B_{MAG,y}$", r"$B_{EAS,y}$"])
axs[1].set_ylim(-1.2,1.2)
axs[1].grid()

axs[2].plot(time_mag,Bz_mag)
axs[2].plot(time_eas,Bz_eas)
axs[2].set_ylabel(r'$B_{z}$'+'\n(normalised)')
axs[2].legend([r"$B_{MAG,z}$", r"$B_{EAS,z}$"])
axs[2].set_ylim(-1.2,1.2)
axs[2].grid()

axs[3].plot(time_mag,B_angle, color='green')
#axs[3].plot(time_eas[:-1],B_angle, color='red') # in case you want to compare the two same series! (one is  currently one point shorter, for some reason)
axs[3].set_ylabel('angle between\nB vectors (°)')
tick_spacing = 2 # degrees
axs[3].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
axs[3].legend([r'$B_{MAG}-B_{EAS}$ Angle'])
axs[3].grid()

axs[Nplots-1].set_xlabel('Date & Time\n(dd hh:mm)')#axs[Nplots-1].set_xlabel('Date & Time (dd hh:mm)')

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