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
    R, theta, phi = vector[0], (vector[1]*np.pi/180)*astro.units.rad, (vector[2]*np.pi/180)*astro.units.rad
    vectorNew = astrocoo.SphericalRepresentation(phi,theta,R).represent_as(astrocoo.CartesianRepresentation)
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


def binIndexFinder(angle, binLowerBoundArray, binUpperBoundArray, binArray, dp=3):
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


def binFinder(vector_sphe, head, bin_dict, dp=3):
    # calculates the parallel and antiparallel elevation and azimuth bins in the EASX head frame for a B vector
    # vector_sphe = B vector in spherical coordinates, in the EASX head frame
    # head = the EASX head frame to use (i.e. the frame in which vector_sphe is measured)
    # bin_dict = bin_dictionary, where all the bin data are stored

    B_elev_bin_parallel = binIndexFinder(vector_sphe[1], bin_dict[head]['ELEVATION_lower_bound'], bin_dict[head]['ELEVATION_upper_bound'], bin_dict[head]['ELEVATION'], dp=dp)
    #B_elev_bin_antiparallel = bin_dict[head]['ELEVATION_bin_count'] - 1 - B_elev_bin_parallel
    B_elev_bin_antiparallel = binIndexFinder(-vector_sphe[1], bin_dict[head]['ELEVATION_lower_bound'], bin_dict[head]['ELEVATION_upper_bound'], bin_dict[head]['ELEVATION'], dp=dp)
    B_elev_bin = np.array([B_elev_bin_parallel, B_elev_bin_antiparallel])

    B_azim_bin_parallel = binIndexFinder(vector_sphe[2], bin_dict[head]['AZIMUTH_lower_bound'], bin_dict[head]['AZIMUTH_upper_bound'], bin_dict[head]['AZIMUTH'], dp=dp)
    #B_azim_bin_antiparallel = (B_azim_bin_parallel + int((bin_dict[head]['AZIMUTH_bin_count'] - 1)/2)) % (bin_dict[head]['AZIMUTH_bin_count'] - 1)
    B_azim_bin_antiparallel = binIndexFinder((vector_sphe[2] + 180) % 360, bin_dict[head]['AZIMUTH_lower_bound'], bin_dict[head]['AZIMUTH_upper_bound'], bin_dict[head]['AZIMUTH'], dp=dp)
    B_azim_bin = np.array([B_azim_bin_parallel, B_azim_bin_antiparallel])

    B_elev = np.array([bin_dict[head]['ELEVATION'][B_elev_bin_parallel], bin_dict[head]['ELEVATION'][B_elev_bin_antiparallel]]) #+1*int(bin_dict[head]['ELEVATION'][B_elev_bin_parallel] >= 0)], bin_dict[head]['ELEVATION'][B_elev_bin_antiparallel]])
    B_azim = np.array([bin_dict[head]['AZIMUTH'][B_azim_bin_parallel], bin_dict[head]['AZIMUTH'][B_azim_bin_antiparallel]])
    
    return B_elev_bin, B_azim_bin, B_elev, B_azim


def binFinder_UpperBound(vector_sphe, head, bin_dict):
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

    B_elev = np.array([bin_dict[head]['ELEVATION_upper_bound'][B_elev_bin_parallel], bin_dict[head]['ELEVATION_upper_bound'][B_elev_bin_antiparallel]]) #+1*int(bin_dict[head]['ELEVATION'][B_elev_bin_parallel] >= 0)], bin_dict[head]['ELEVATION'][B_elev_bin_antiparallel]])
    B_azim = np.array([bin_dict[head]['AZIMUTH_upper_bound'][B_azim_bin_parallel], bin_dict[head]['AZIMUTH_upper_bound'][B_azim_bin_antiparallel]])
    
    return B_elev_bin, B_azim_bin, B_elev, B_azim


def binFinder_LowerBound(vector_sphe, head, bin_dict):
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

    B_elev = np.array([bin_dict[head]['ELEVATION_lower_bound'][B_elev_bin_parallel], bin_dict[head]['ELEVATION_lower_bound'][B_elev_bin_antiparallel]]) #+1*int(bin_dict[head]['ELEVATION'][B_elev_bin_parallel] >= 0)], bin_dict[head]['ELEVATION'][B_elev_bin_antiparallel]])
    B_azim = np.array([bin_dict[head]['AZIMUTH_lower_bound'][B_azim_bin_parallel], bin_dict[head]['AZIMUTH_lower_bound'][B_azim_bin_antiparallel]])
    
    return B_elev_bin, B_azim_bin, B_elev, B_azim


def getEASXBinProjections(head,bin_dict,cart_proj_tuple=(1,1),point_density=10):
    # projects EASX bins from EASX spherical coordinates to other spherical coordinates
    # head = the head in question (0 (EAS1) or 1 (EAS2))
    # bin_dict = dictionary with all bin boundary data
    # cart_proj_tuple = tuple with two cartesian coordinate transform matrices to desired projection. first for EAS1, second for EAS2.
    # point_density = how many points to plot per degree

    elevBinCount = bin_dict[head]['ELEVATION_bin_count']
    azimBinCount = bin_dict[head]['AZIMUTH_bin_count']
    
    elevLowerBoundArray = bin_dict[head]['ELEVATION_lower_bound']
    elevUpperBoundArray = bin_dict[head]['ELEVATION_upper_bound']
    azimLowerBoundArray = bin_dict[head]['AZIMUTH_lower_bound']
    azimUpperBoundArray = bin_dict[head]['AZIMUTH_upper_bound']

    elevDeltaLowerArray = bin_dict[head]['ELEVATION_delta_lower']

    azimuthPointCount = int(point_density*360/azimBinCount)
    azimuthPointStep = 1/point_density

    elevationPointStep = 1/point_density

    #binBoundaryProjectionArray = np.ndarray((elevBinCount,azimBinCount)) # 2D array to store the projected points along the boundary of each bin
    binBoundaryProjectionArray = []
    print('\nbin projection:')
    for i in range(elevBinCount):
        elevationBinWidth = 2*elevDeltaLowerArray[i]
        elevationPointCount = int(point_density*elevationBinWidth)
        binBoundaryProjectionArray.append([])
        for j in range(azimBinCount):
            pointArray = [] # array of points around a single elevzimuth bin/"pixel"
            for k_az in range(0,azimuthPointCount):
                lowerEdge_elev = np.array([1, elevLowerBoundArray[i], azimLowerBoundArray[j]+k_az*azimuthPointStep]) # vary azimuth along lower elevation
                lowerEdge_elev = cartToSphere(cart_proj_tuple[head].dot(sphereToCart(lowerEdge_elev))) # EASX sphe -> EASX cart -> SRF cart -> SRF sphe
                pointArray.append(lowerEdge_elev)

                upperEdge_elev = np.array([1, elevUpperBoundArray[i], azimLowerBoundArray[j]+k_az*azimuthPointStep]) # vary azimuth along upper elevation
                upperEdge_elev = cartToSphere(cart_proj_tuple[head].dot(sphereToCart(upperEdge_elev)))
                pointArray.append(upperEdge_elev)

            for k_el in range(0,elevationPointCount):
                lowerEdge_azim = np.array([1, elevLowerBoundArray[i]+k_el*elevationPointStep, azimLowerBoundArray[j]]) # vary elevation along lower azimuth
                lowerEdge_azim = cartToSphere(cart_proj_tuple[head].dot(sphereToCart(lowerEdge_azim)))
                pointArray.append(lowerEdge_azim)

                upperEdge_azim = np.array([1, elevLowerBoundArray[i]+k_el*elevationPointStep, azimUpperBoundArray[j]]) # vary elevation along upper azimuth
                upperEdge_azim = cartToSphere(cart_proj_tuple[head].dot(sphereToCart(upperEdge_azim)))
                pointArray.append(upperEdge_azim)

            for point in pointArray:
                point[2] = (point[2]) - 360*(point[2] > 180) # convert azimuth from [0,360) to (-180,180]
            binBoundaryProjectionArray[i].append(np.array(pointArray).T)

            print('\relevation = {}, azimuth = {}'.format(i, j), end='')
    print('')

    return binBoundaryProjectionArray


def getEASXVectorProjections(head,vector=np.array([0,0,0]),pitch_contours=[],cart_proj_tuple=(1,1),point_density=10):
    # projects vector from EASX spherical coordinates to other spherical coordinates. Adapted from Chris Owen's algorithm.
    # head = the head in question (0 (EAS1) or 1 (EAS2))
    # vector = vector in EASX spherical coordinates
    # pitch_contours = optional list of pitch angles in degrees for plotting a line of constant pitch angle relative to the vector
    # cart_proj_tuple = tuple with two cartesian coordinate transform matrices to desired projection. first for EAS1, second for EAS2.
    # point_density = how many points to plot per degree
    
    radius, el, az = vector[0], vector[1], vector[2]
    
    vector = cartToSphere(cart_proj_tuple[head].dot(sphereToCart(vector))) # EASX sphe -> EASX cart -> SRF cart -> SRF sphe
    vector[2] = (vector[2]) - 360*(vector[2] > 180)

    antivector = np.array([radius,-el,(az+180)%360])
    antivector = cartToSphere(cart_proj_tuple[head].dot(sphereToCart(antivector)))
    antivector[2] = (antivector[2]) - 360*(antivector[2] > 180)

    theta, phi = vector[1]*np.pi/180, vector[2]*np.pi/180

    # Let's use Chris's algorithm!
    Mrot = np.array([[np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), np.sin(theta)],
                    [-np.sin(phi), np.cos(phi), 0],
                    [-np.cos(phi)*np.sin(theta), -np.sin(phi)*np.sin(theta), np.cos(theta)]]) # transform matrix to the frame defined by the field vector
    Mrot_inv = np.linalg.inv(Mrot)

    contourArray = []
    print('\nvector contour projection:')
    for pitch in pitch_contours:
        phaseCount = int(2*np.pi*abs(np.sin(pitch*np.pi/180))*90*point_density) # phasecount is given by a rough estimate of the "circumference" of the contour, in degrees
        phaseArray = np.linspace(0,2*np.pi,phaseCount) # array of phase angles to go around the contour
        pitch = pitch * np.pi/180
        sinpitch = np.sin(pitch)
        cospitch = np.cos(pitch)
        for i in range(phaseCount):
            phase = phaseArray[i]
            conVec_cart = np.array([cospitch, sinpitch*np.cos(phase), sinpitch*np.sin(phase)])
            conVec_cart_fieldframe = Mrot_inv.dot(conVec_cart)
            conPhi = np.arctan2(conVec_cart_fieldframe[1],(conVec_cart_fieldframe[0]))*180/np.pi
            conTheta = np.arctan2(conVec_cart_fieldframe[2],(np.sqrt(conVec_cart_fieldframe[0]**2+conVec_cart_fieldframe[1]**2)))*180/np.pi
            contourArray.append(np.array([radius, conTheta, conPhi]))
            print('\rpitch angle = {}, {:.0f}% complete'.format(int(pitch*180/np.pi), 100*i/phaseCount), end='')
    print('')
    
    # contourArray = []
    # print('\nvector contour projection:')
    # for pitch in pitch_contours:
    #     ampSin = np.sin(pitch*np.pi/180)
    #     ampCos = np.cos(pitch*np.pi/180)
    #     pointCount = 2*np.pi*pitch*point_density
    #     pointCount = int(pointCount)
    #     angleStep = 2*np.pi/pointCount
    #     cosPitch = np.cos(pitch*np.pi/180)
    #     sinPitch = np.sin(pitch*np.pi/180)
    #     v = np.array(np.cos(pitch)/kx)
    #     for i in range(pointCount):
    #         el2 = i*angleStep
    #         contourVector = np.array([radius, el + (np.sin(el2))*pitch, az + (np.cos(el2))*pitch])
    #         contourVector[1] = (contourVector[1]) - 180*(contourVector[1] > 90) + 180*(contourVector[1] < -90) # keep elev between +/-90
    #         contourVector = cartToSphere(cart_proj_tuple[head].dot(sphereToCart(contourVector)))
    #         contourVector[1] = (contourVector[1]) - 90*(contourVector[1] > 90) + 90*(contourVector[1] < -90) # keep elev between +/-90
    #         contourVector[2] = (contourVector[2]) - 360*(contourVector[2] > 180) # convert azimuth from [0,360) to (-180,180]
    #         contourArray.append(contourVector)
    #         print('\rpitch angle = {}, {:.0f}% complete'.format(pitch, 100*i/pointCount), end='')
    # print('')

    return vector, antivector, np.array(contourArray).T


def getBinDictsFromBounds(EASXtop=[[],[]], EASXbot=[[],[]]):
    # gets you a tuple of bin_dicts if you only have the top and bottom bounds of the bins for EASX, assuming azimuth bins are unchanged
    # EASXtop = list of top bound arrays = (EAS1top,EAS2top)
    # EASXbot = list of bottom bound arrays = (EAS1bot,EAS2bot)

    dictionaries = [{},{}]

    for i in range(2):
        dictionary = {'ELEVATION':np.ndarray(16),
        'ELEVATION_delta_lower':np.ndarray(16),
        'ELEVATION_delta_upper':np.ndarray(16),
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}

        dictionary['ELEVATION_lower_bound'] = EASXbot[i]
        dictionary['AZIMUTH_lower_bound'] = dictionary['AZIMUTH'] - dictionary['AZIMUTH_delta_lower']
        dictionary['ELEVATION_upper_bound'] = EASXtop[i]
        dictionary['AZIMUTH_upper_bound'] = dictionary['AZIMUTH'] + dictionary['AZIMUTH_delta_upper']
        dictionary['ELEVATION_bin_count'] = len(dictionary['ELEVATION'])
        dictionary['AZIMUTH_bin_count'] = len(dictionary['AZIMUTH'])

        for j in range(len(dictionary['ELEVATION_delta_lower'])):
                delta = (dictionary['ELEVATION_upper_bound'][j] - dictionary['ELEVATION_lower_bound'][j])/2
                dictionary['ELEVATION_delta_lower'][j] = delta
                dictionary['ELEVATION_delta_upper'][j] = delta
                dictionary['ELEVATION'][j] = dictionary['ELEVATION_upper_bound'][j] + delta
        
        dictionaries[i] = dictionary

    return (dictionaries[0],dictionaries[1])


def getAngleLoss(head,bin_dict,B_mag=[[0,0,0],[0,0,0]],bin_indices=[0,0],dp=3):
    # tells you which angles are lost due to a discrepancy between B_mag and the bin chosen by B_eas
    # head = head used
    # bin_dict = dictionary with binning data
    # B_mag = pair for the parallel and antiparallel true ground vector in spherical coordinates in the head used by eas
    # bin_indices = pair of indices for the parallel and antiparallel B_eas elevation bin chosen
    
    bin_indices = list(map(int, bin_indices))

    losses = [0,0]
    for i in range(2):
        radius, el, az = B_mag[i][0], B_mag[i][1], B_mag[i][2]
        binLowerBound = bin_dict[head]['ELEVATION_lower_bound'][bin_indices[i]]
        binUpperBound = bin_dict[head]['ELEVATION_upper_bound'][bin_indices[i]]

        # print('lower: ' + str(binLowerBound))
        # print('el: ' + str(el))
        # print('upper: ' + str(binUpperBound))

        angleLoss = 0 
        if not (binLowerBound <= el and el <= binUpperBound): # if B_mag is not inside the same elevation bin, then:
            angle1 = abs(el - binLowerBound)
            angle2 = abs(el - binUpperBound)
            if angle1 < angle2: # what's the closest the vector reaches to the bin in terms of elevation?
                angleLoss = angle1
            else:
                angleLoss = angle2
        # else:
        #     print('no loss!')

        losses[i] = angleLoss

    return losses


def plotBinProjections(ax,headToRead,el_selected=[],az_selected=[],pix_selected=[],bin_color='b',el_color='r',az_color='g',pix_color='y',s=0.1):
    # read and plot all projected bins in blue
    # ax = axis to plot to
    # headToRead = 0 for EAS1, 1 for EAS2
    # el_selected = array of indices of elevation bins to highlight
    # az_selected = array of indices of azimuth bins to highlight
    # pix_selected = 2d array of index coordinates [el, az] of elevzimuth "pixels" to highlight
    # s = point size
    
    # read and plot all bins in the selected head:
    headToRead = 0
    print('\nreading bins:')
    for i in range(16):
            for j in range(32):
                    filename = 'bin_projections\EAS{}\el{}\EAS{}_el{}_az{}.csv'.format(headToRead+1, i, headToRead+1, i, j)
                    pixelArray = np.genfromtxt(filename, delimiter=",")
                    el = pixelArray[1]
                    az = pixelArray[2]
                    ax.scatter(az,el,color=bin_color,marker='.',s=s)
                    print('\relevation = {}, azimuth = {}'.format(i, j), end='')
    print('')

    # read and plot the selected elevation bins:
    for el in el_selected:
        for j in range(32):
                        filename = 'bin_projections\EAS{}\el{}\EAS{}_el{}_az{}.csv'.format(headToRead+1, el, headToRead+1, el, j)
                        pixelArray = np.genfromtxt(filename, delimiter=",")
                        el_plot = pixelArray[1]
                        az_plot = pixelArray[2]
                        ax.scatter(az_plot,el_plot,color=el_color,marker='.',s=s*1.5)

    # read and plot the selected azimuth bins:
    for az in az_selected:
        for i in range(16):
                        filename = 'bin_projections\EAS{}\el{}\EAS{}_el{}_az{}.csv'.format(headToRead+1, i, headToRead+1, i, az)
                        pixelArray = np.genfromtxt(filename, delimiter=",")
                        el_plot = pixelArray[1]
                        az_plot = pixelArray[2]
                        ax.scatter(az_plot,el_plot,color=az_color,marker='.',s=s*1.5)

    # read and plot the selected elevzimuth pixels:
    for pix in pix_selected:
        el, az = pix[0], pix[1]
        filename = 'bin_projections\EAS{}\el{}\EAS{}_el{}_az{}.csv'.format(headToRead+1, el, headToRead+1, el, az)
        pixelArray = np.genfromtxt(filename, delimiter=",")
        el_plot = pixelArray[1]
        az_plot = pixelArray[2]
        ax.scatter(az_plot,el_plot,color=pix_color,marker='.',s=s*1.5)


def gifmaker(directory,gifname,duration=0.125,loop=0):
    # generates a high quality gif given a directory where all your images (and only your images) are saved, and saves it in that same directory. The gif must be deleted externally before running this code again.
    # directory = path to directory where images are saved
    # gifname = what to name the gif
    # duration = duration between each frame, in seconds
    # loop = number of loops. 0 implies infinite loops.
    
    gif_path = directory + "\\" + gifname + ".gif"
    frameslist = os.listdir(directory)
    frames = np.stack([iio.imread(directory + f"\{framename}") for framename in frameslist], axis=0)
    iio.imwrite(gif_path, frames, duration=duration, loop=loop)