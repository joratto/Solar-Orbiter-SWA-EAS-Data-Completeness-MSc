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
    length = len(timeArray)
    time_index = 0
    for i in range(1,searchdivisions+1):
        time_index += int(length/2**i)*(timeArray[time_index+int(length/2**i)] < timeReference) # add smaller and smaller slices to converge on the right time index for t0
    while timeArray[time_index] < timeReference:
        time_index += 1
        print('\rtime index = {}/{}'.format(time_index,length), end='')
        continue

    return time_index


def cropTimeToOverlap(seriesA,timeA,seriesB,timeB,searchdivisions=5):
    # crops two time series to the period where they overlap in time

    if len(timeA) != len(seriesA) or len(timeB) != len(seriesB):
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
        seriesB = seriesB[time_index_t0:]
    
    else: # vice versa.
        t0 = t0B
        print('overlap starts at {}'.format(t0))
        time_index_t0 = cropTime_indexFinder(timeA, t0, searchdivisions)-1
        timeA = timeA[time_index_t0:]
        seriesA = seriesA[time_index_t0:]

    print('\n')
    print('tfA = {}'.format(tfA))
    print('tfB = {}'.format(tfB))

    if tfA > tfB: # if tfA is later than tfB, finish at tfB, therefore series A finishes late and must be cropped.
        tf = tfB
        print('overlap finishes at {}'.format(tf))
        time_index_t0 = cropTime_indexFinder(timeA, tf, searchdivisions)
        timeA = timeA[:time_index_t0]
        seriesA = seriesA[:time_index_t0]
        
    else: # vice versa.
        tf = tfA
        print('overlap finishes at {}'.format(tf))
        time_index_t0 = cropTime_indexFinder(timeB, tf, searchdivisions)
        timeB = timeB[:time_index_t0]
        seriesB = seriesB[:time_index_t0]

    print('\n')
    print('\noverlap duration = {}'.format(tf-t0))
    print('\n')

    return seriesA, timeA, seriesB, timeB


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
    vectorBNew = np.array([abs(Bx),By,Bz]) # to make sure we get the shortes angle to each z axis
    EAS1z_angle = np.arccos(np.clip(np.dot(vectorBNew,EAS1zAxis),-1,1))*180/np.pi # shortest angle in degrees
    EAS2z_angle = np.arccos(np.clip(np.dot(vectorBNew,EAS2zAxis),-1,1))*180/np.pi # shortest angle in degrees
    #print('{}, {}'.format(B_mag_EAS1z_angle, B_mag_EAS2z_angle))
    head = 0  # 0 indicates EAS1, 1 indicates EAS2
    if EAS2z_angle > EAS1z_angle: # pick the head with the largest angle
        head = 1
    
    return head, EAS1z_angle, EAS2z_angle


def binFinder(angle, binLowerBoundArray, binUpperBoundArray, dp=3):
    # calculates the index of the bin that an angle falls into
    # angle = angle in degrees
    # binLowerBoundArray = array of lower bounds of bins
    # binUpperBoundArray = array of upper bounds of bins
    # decimal points = number of decimal points for the lower and upper bounds to avoid contradictions that pop up
    binCount = len(binLowerBoundArray)
    binIndex = 0
    while not ((round(binLowerBoundArray[binIndex], dp) < round(angle, dp)) and (round(angle, dp) < round(binUpperBoundArray[binIndex], dp))):
        binIndex += 1
        if binIndex+1 == binCount:
            '''angle out of range!'''
            break
    # note: the "round" function behaves unusually, e.g. rounding 22.885 to 22.88 (see https://docs.python.org/2/library/functions.html#round). This should only affect elevation bin 12 (-22.885 vs -22.89), and the angle difference is minuscule anyway.

    return binIndex