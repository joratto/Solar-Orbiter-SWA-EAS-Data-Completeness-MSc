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
    vectorNew = astrocoo.CartesianRepresentation(x,y,z).represent_as(astrocoo.SphericalRepresentation)
    R, theta, phi = vectorNew.distance.value, vectorNew.lat.value*180/np.pi, vectorNew.lon.value*180/np.pi
    
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
