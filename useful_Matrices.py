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
dictionary = EAS1_bin_dict
dictionary['ELEVATION_lower_bound'] = dictionary['ELEVATION'] - dictionary['ELEVATION_delta_lower'] # subtract elevation bin lower deltas from elevation bin centers to get the lower bounds
dictionary['AZIMUTH_lower_bound'] = dictionary['AZIMUTH'] - dictionary['AZIMUTH_delta_lower']
dictionary['ELEVATION_upper_bound'] = dictionary['ELEVATION'] + dictionary['ELEVATION_delta_upper'] # add elevation bin upper deltas to elevation bin centers to get the upper bounds
dictionary['AZIMUTH_upper_bound'] = dictionary['AZIMUTH'] + dictionary['AZIMUTH_delta_upper']
dictionary['ELEVATION_bin_count'] = len(dictionary['ELEVATION'])
dictionary['AZIMUTH_bin_count'] = len(dictionary['AZIMUTH'])

EAS2_bin_dict = {'ELEVATION':np.array([38.94,  28.25,  19.86,  12.99,   7.25,   2.35,  -1.93,  -5.78,  -9.37, -12.84, -16.32, -19.97, -23.97, -28.57, -34.13, -41.1]),
        'ELEVATION_delta_lower':np.array([6.06,  4.633, 3.761, 3.111, 2.624, 2.272, 2.012, 1.838, 1.747, 1.722, 1.759, 1.887, 2.113, 2.485, 3.071, 3.897]),
        'ELEVATION_delta_upper':np.array([6.06,  4.633, 3.761, 3.111, 2.624, 2.272, 2.012, 1.838, 1.747, 1.722, 1.759, 1.887, 2.113, 2.485, 3.071, 3.897]),
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}
dictionary = EAS2_bin_dict
dictionary['ELEVATION_lower_bound'] = dictionary['ELEVATION'] - dictionary['ELEVATION_delta_lower'] # subtract elevation bin lower deltas from elevation bin centers to get the lower bounds
dictionary['AZIMUTH_lower_bound'] = dictionary['AZIMUTH'] - dictionary['AZIMUTH_delta_lower']
dictionary['ELEVATION_upper_bound'] = dictionary['ELEVATION'] + dictionary['ELEVATION_delta_upper'] # add elevation bin upper deltas to elevation bin centers to get the upper bounds
dictionary['AZIMUTH_upper_bound'] = dictionary['AZIMUTH'] + dictionary['AZIMUTH_delta_upper']
dictionary['ELEVATION_bin_count'] = len(dictionary['ELEVATION'])
dictionary['AZIMUTH_bin_count'] = len(dictionary['AZIMUTH'])

bin_dictionary = (EAS1_bin_dict, EAS2_bin_dict) # the bin data for the two EAS heads


EAS2_bin_dict_old = {'ELEVATION':np.array([ 39.044788,    28.411247,    19.576036,    11.764658,     5.0754714, -0.49025512,  -5.1967564,   -9.1976643,  -12.682464,   -15.899632,  -18.874838,   -21.873714,   -25.295139,   -29.432489,   -34.743786, -41.378841]),
        'ELEVATION_delta_lower':np.array([5.9548,   4.6812,   4.156,    3.6547,   3.03547,  2.529745, 2.17324,  1.82234,  1.6575,   1.5604,   1.4152,   1.5863,   1.8449,   2.2975,   3.0162,   3.6212 ]),
        'ELEVATION_delta_upper':np.array([5.9552,   4.6788,   4.154,    3.6553,   3.03453,  2.530255, 2.17676,  1.82766, 1.6625,   1.5596,   1.4148,   1.5837,   1.8351,   2.2925,   3.0138,   3.6188  ]),
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}
dictionary = EAS2_bin_dict_old
dictionary['ELEVATION_lower_bound'] = dictionary['ELEVATION'] - dictionary['ELEVATION_delta_lower'] # subtract elevation bin lower deltas from elevation bin centers to get the lower bounds
dictionary['AZIMUTH_lower_bound'] = dictionary['AZIMUTH'] - dictionary['AZIMUTH_delta_lower']
dictionary['ELEVATION_upper_bound'] = dictionary['ELEVATION'] + dictionary['ELEVATION_delta_upper'] # add elevation bin upper deltas to elevation bin centers to get the upper bounds
dictionary['AZIMUTH_upper_bound'] = dictionary['AZIMUTH'] + dictionary['AZIMUTH_delta_upper']
dictionary['ELEVATION_bin_count'] = len(dictionary['ELEVATION'])
dictionary['AZIMUTH_bin_count'] = len(dictionary['AZIMUTH'])

EAS1_bin_dict_old = {'ELEVATION':np.array([ 39.530624,    29.387497,    20.671091,    13.129827,     6.478693, 0.60467362,  -4.5230989,   -8.9728432,  -12.822032,   -16.335674, -19.694551,   -22.974125,   -26.428017,   -30.432095,   -35.349419, -41.543976  ]),
        'ELEVATION_delta_lower':np.array([5.4706,   4.6775,   4.0411,   3.4998,   3.14869,  2.724674, 2.4069,   2.04716, 1.808,    1.7043,   1.6554,   1.6259,   1.822,    2.1779,   2.7406,   3.456   ]),
        'ELEVATION_delta_upper':np.array([5.4694,   4.6725,   4.0389,   3.5002,   3.15131,  2.725326, 2.4031,   2.04284, 1.802,    1.7057,   1.6546,   1.6241,   1.828,    2.1821,   2.7394,   3.454   ]),
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}
dictionary = EAS1_bin_dict_old
dictionary['ELEVATION_lower_bound'] = dictionary['ELEVATION'] - dictionary['ELEVATION_delta_lower'] # subtract elevation bin lower deltas from elevation bin centers to get the lower bounds
dictionary['AZIMUTH_lower_bound'] = dictionary['AZIMUTH'] - dictionary['AZIMUTH_delta_lower']
dictionary['ELEVATION_upper_bound'] = dictionary['ELEVATION'] + dictionary['ELEVATION_delta_upper'] # add elevation bin upper deltas to elevation bin centers to get the upper bounds
dictionary['AZIMUTH_upper_bound'] = dictionary['AZIMUTH'] + dictionary['AZIMUTH_delta_upper']
dictionary['ELEVATION_bin_count'] = len(dictionary['ELEVATION'])
dictionary['AZIMUTH_bin_count'] = len(dictionary['AZIMUTH'])

old_bin_dictionary = (EAS1_bin_dict_old, EAS2_bin_dict_old)

#print(EAS1_bin_dict['ELEVATION_lower_bound'])
#print(EAS1_bin_dict['ELEVATION'])
#print(EAS1_bin_dict['ELEVATION_upper_bound'])
#print(EAS1_bin_dict['AZIMUTH_lower_bound'])
#print(EAS1_bin_dict['AZIMUTH'])
#print(EAS1_bin_dict['AZIMUTH_upper_bound'])
#print('HOOOLD IIIIT')