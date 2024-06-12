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


def cropTime_indexFinder(timeArray,timeReference,searchdivisions=5):
    # this function finds a time index given a time, efficiently.
    # timeArray is the time array where you're finding an index.
    # timeReference is the reference time whose index you're trying to find.
    # searchdivisions is the number of times timeArray should be divided to narrow the search space.

    #timeReference = timeReference.replace(microsecond=0)
    length = len(timeArray)
    time_index = 0
    for i in range(1,searchdivisions+1):
        time_index += int(length/2**i)*(timeArray[time_index+int(length/2**i)] < timeReference) # add smaller and smaller slices to converge on the right time index for t0
    print('')
    while (timeArray[time_index] < timeReference) and (time_index < length-1):
        time_index += 1
        print('\rtime index = {}/{}'.format(time_index,length), end='')
        continue
    print('')

    return time_index


def cropTimeToRef(seriesArray,time,timeRef_0,timeRef_f,searchdivisions=5):
    # crops a set of time series with the same time to a specific time reference
    # seriesArray = an array of equally sized time series, all of which correspond to time
    # time = the time axis for each time series in seriesArray
    # timeRef_0 is the starting reference time you're trying to crop to
    # timeRef_f is the end reference time you're trying to crop to
    
    seriesCount = len(seriesArray)
    time_index_t0 = cropTime_indexFinder(time, timeRef_0, searchdivisions) # "-1" makes sure the longer/later series isn't off by 1 (1 point longer), and that exactly B_eas has has exactly 7200 vectors
    time_index_tf = cropTime_indexFinder(time, timeRef_f, searchdivisions)
    time = time[time_index_t0:time_index_tf]
    for i in range(seriesCount):
        seriesArray[i] = seriesArray[i][time_index_t0:time_index_tf]
    
    return seriesArray, time


def getFilenameDatetime_EAS(filename=''):
    fields = filename.split("_")
    times = fields[3].split("-")
    year =   (times[0][ 0:4 ], times[1][ 0:4 ])
    month =  (times[0][ 4:6 ], times[1][ 4:6 ])
    day =    (times[0][ 6:8 ], times[1][ 6:8 ])
    hour =   (times[0][ 9:11], times[1][ 9:11])
    minute = (times[0][11:13], times[1][11:13])
    second = (times[0][13:15], times[1][13:15])
    t0 = datetime.datetime(year=int(year[0]),month=int(month[0]),day=int(day[0]),hour=int(hour[0]),minute=int(minute[0]),second=int(second[0]))
    tf = datetime.datetime(year=int(year[1]),month=int(month[1]),day=int(day[1]),hour=int(hour[1]),minute=int(minute[1]),second=int(second[1]))

    return t0, tf


def cropTimeToOverlap(seriesA_Array,timeA,seriesB_Array,timeB,searchdivisions=5):
    # crops two sets of time series with different durations to the period where they overlap in time
    # seriesA_Array = an array of equally sized time series, all of which correspond to timeA
    # timeA = the time axis for each time series in seriesA_Array

    timeA_Length = len(timeA)
    timeB_Length = len(timeB)

    seriesA_Count = len(seriesA_Array)
    seriesB_Count = len(seriesB_Array)

    for seriesA in seriesA_Array:
            if timeA_Length != len(seriesA):
                raise Exception('all time and series axes must be the same size!')
    for seriesB in seriesB_Array:
            if timeB_Length != len(seriesB):
                raise Exception('all time and series axes must be the same size!')

    t0A = timeA[0] # timeA start
    t0B = timeB[0] # timeB start
    tfA = timeA[-1] # timeA finish
    tfB = timeB[-1] # timeB finish

    t0 = 0 # overlap start
    tf = 0 # overlap finish

    if t0A > tfB or t0B > tfA:
        raise Exception('these time series do not overlap!')

    print('\n')
    print('t0A = {}'.format(t0A))
    print('t0B = {}'.format(t0B))

    if t0A > t0B: # if t0A is later than t0B, start at t0A, therefore series B starts early and must be cropped.
        t0 = t0A
        print('overlap starts at {}'.format(t0))
        time_index_t0 = cropTime_indexFinder(timeB, t0, searchdivisions)-1 # "-1" makes sure the longer/later series isn't off by 1 (1 point longer), and that exactly B_eas has has exactly 7200 vectors
        timeB = timeB[time_index_t0:]
        for i in range(seriesB_Count):
            seriesB_Array[i] = seriesB_Array[i][time_index_t0:]
    
    else: # vice versa.
        t0 = t0B
        print('overlap starts at {}'.format(t0))
        time_index_t0 = cropTime_indexFinder(timeA, t0, searchdivisions)-1
        timeA = timeA[time_index_t0:]
        for i in range(seriesA_Count):
            seriesA_Array[i] = seriesA_Array[i][time_index_t0:]

    print('\n')
    print('tfA = {}'.format(tfA))
    print('tfB = {}'.format(tfB))

    if tfA > tfB: # if tfA is later than tfB, finish at tfB, therefore series A finishes late and must be cropped.
        tf = tfB
        print('overlap finishes at {}'.format(tf))
        time_index_tf = cropTime_indexFinder(timeA, tf, searchdivisions)
        timeA = timeA[:time_index_tf]
        for i in range(seriesA_Count):
            seriesA_Array[i] = seriesA_Array[i][:time_index_tf]
        
    else: # vice versa.
        tf = tfA
        print('overlap finishes at {}'.format(tf))
        time_index_tf = cropTime_indexFinder(timeB, tf, searchdivisions)
        timeB = timeB[:time_index_tf]
        for i in range(seriesB_Count):
            seriesB_Array[i] = seriesB_Array[i][:time_index_tf]

    print('\n')
    print('\noverlap duration = {}'.format(tf-t0))
    print('\n')

    return seriesA_Array, timeA, seriesB_Array, timeB


def cartToSphere(vector):
    # input should be cartesian (x,y,z)
    x, y, z, = vector[0], vector[1], vector[2]
    vectorNew = astrocoo.CartesianRepresentation(x,y,z).represent_as(astrocoo.SphericalRepresentation)
    R, theta, phi = vectorNew.distance.value, vectorNew.lat.value*180/np.pi, vectorNew.lon.value*180/np.pi
    # output is spherical, in degrees
    return np.array([R,theta,phi])


def sphereToCart(vector):
    # input should be spherical, in degrees (R,theta,phi)
    R, theta, phi = vector[0], vector[1]*np.pi/180, vector[2]*np.pi/180
    vectorNew = astrocoo.SphericalRepresentation(R,theta,phi).represent_as(astrocoo.CartesianRepresentation)
    x, y, z = vectorNew.x.value, vectorNew.y.value, vectorNew.z.value
    # output is cartesian
    return np.array([x,y,z])


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


def headPicker(vectorB,EAS1zAxis,EAS2zAxis):
    # calculate angles between a B vector and the EAS1 z-axis and B vector and the EAS2 z-axis to pick the EAS sensor head such that the B vector lies closest to its aperture center plane (as per Owen et al 2021).
    # vectorB = unit magnetic field vector
    # EAS1zAxis = unit vector along EAS1 z-axis
    # EAS2zAxis = unit vector along EAS2 z-axis
    Bx, By, Bz = vectorB[0], vectorB[1], vectorB[2]
    vectorBNew = np.array([Bx,By,Bz]) # to make sure we get the shortes angle to each z axis
    EAS1z_angle = np.arccos(np.clip(np.dot(vectorBNew,EAS1zAxis),-1,1))*180/np.pi # shortest angle in degrees
    EAS2z_angle = np.arccos(np.clip(np.dot(vectorBNew,EAS2zAxis),-1,1))*180/np.pi # shortest angle in degrees
    #print('{}, {}'.format(B_mag_EAS1z_angle, B_mag_EAS2z_angle))
    head = 0  # 0 indicates EAS1, 1 indicates EAS2
    if abs(abs(EAS2z_angle)-90) <= abs(abs(EAS1z_angle)-90): # pick the head with the smallest angle between B and the aperture midplane
        head = 1
    
    return head, EAS1z_angle, EAS2z_angle


def binIndexFinder(angle, binLowerBoundArray, binUpperBoundArray, binArray, dp=0):
    # calculates the index of the bin that an angle falls into
    # angle = angle in degrees
    # binLowerBoundArray = array of lower bounds of bins
    # binUpperBoundArray = array of upper bounds of bins
    # binArray = array of centers of bins
    # decimal points = number of decimal points for the lower and upper bounds to avoid contradictions that pop up

    #binLowerBoundArray, binUpperBoundArray, binArray = np.flip(binLowerBoundArray*1), np.flip(binUpperBoundArray*1), np.flip(binArray*-1)

    binCount = len(binLowerBoundArray)
    binMin = round(min(binLowerBoundArray), dp) # minimum value in binLowerBoundArray
    binMax = round(max(binUpperBoundArray), dp) # maximum value in binUpperBoundArray
    angle = round(np.clip(angle, binMin, binMax), dp) # clip the angle to within the bin range, and round to dp
    binIndex = 0
    for i in range(binCount):
        if (((round(binLowerBoundArray[binIndex], dp) <= angle) and (angle <= round(binUpperBoundArray[binIndex], dp)))) or (binIndex+1 == binCount):
            break
        binIndex += 1
    
    # binArray = sorted(binArray)
    # if not (((binMax <= angle) and (angle <= round(binArray[binIndex], dp)) or ((round(binArray[binIndex], dp) <= angle) and (angle <= binMin)))):
    #     for i in range(0,binCount-1):
    #         if ((round(binArray[binIndex], dp) <= angle) and (angle <= round(binArray[binIndex+1], dp))) or (binIndex+1 == binCount):
    #             break
    #         binIndex += 1
    # binIndex = binCount - 1 - binIndex

    # while not ((round(binLowerBoundArray[binIndex], dp) < angle) and (angle < round(binUpperBoundArray[binIndex], dp))):
    #     binIndex += 1
    #     #if binIndex+1 == binCount:
    #     #    '''angle out of range!'''
    #     #    break
    # note: the "round" function behaves unusually, e.g. rounding 22.885 to 22.88 (see https://docs.python.org/2/library/functions.html#round). This should only affect elevation bin 12 (-22.885 vs -22.89), and the angle difference is minuscule anyway.

    return binIndex


def binFinder(vector_sphe, head, bin_dict):
    # calculates the parallel and antiparallel elevation and azimuth bins in the EASX head frame for a B vector
    # vector_sphe = B vector in spherical coordinates, in the EASX head frame
    # head = the EASX head frame to use (i.e. the frame in which vector_sphe is measured)
    # bin_dict = bin_dictionary, where all the bin data are stored

    B_elev_bin_parallel = binIndexFinder(vector_sphe[1], bin_dict[head]['ELEVATION_lower_bound'], bin_dict[head]['ELEVATION_upper_bound'], bin_dict[head]['ELEVATION'])
    #B_elev_bin_antiparallel = bin_dict[head]['ELEVATION_bin_count'] - 1 - B_elev_bin_parallel
    B_elev_bin_antiparallel = binIndexFinder(-vector_sphe[1], bin_dict[head]['ELEVATION_lower_bound'], bin_dict[head]['ELEVATION_upper_bound'], bin_dict[head]['ELEVATION'])
    B_elev_bin = np.array([B_elev_bin_parallel, B_elev_bin_antiparallel])

    B_azim_bin_parallel = binIndexFinder(vector_sphe[2], bin_dict[head]['AZIMUTH_lower_bound'], bin_dict[head]['AZIMUTH_upper_bound'], bin_dict[head]['AZIMUTH'])
    #B_azim_bin_antiparallel = (B_azim_bin_parallel + int((bin_dict[head]['AZIMUTH_bin_count'] - 1)/2)) % (bin_dict[head]['AZIMUTH_bin_count'] - 1)
    B_azim_bin_antiparallel = binIndexFinder((vector_sphe[2] + 180) % 360, bin_dict[head]['AZIMUTH_lower_bound'], bin_dict[head]['AZIMUTH_upper_bound'], bin_dict[head]['AZIMUTH'])
    B_azim_bin = np.array([B_azim_bin_parallel, B_azim_bin_antiparallel])

    B_elev = np.array([bin_dict[head]['ELEVATION'][B_elev_bin_parallel], bin_dict[head]['ELEVATION'][B_elev_bin_antiparallel]]) #+1*int(bin_dict[head]['ELEVATION'][B_elev_bin_parallel] >= 0)], bin_dict[head]['ELEVATION'][B_elev_bin_antiparallel]])
    B_azim = np.array([bin_dict[head]['AZIMUTH'][B_azim_bin_parallel], bin_dict[head]['AZIMUTH'][B_azim_bin_antiparallel]])
    
    return B_elev_bin, B_azim_bin, B_elev, B_azim