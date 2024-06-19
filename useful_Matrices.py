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



EAS2_bin_dict_lessold = {'ELEVATION':np.array([39.906, 29.466, 20.979, 13.887, 7.848, 2.601, -2.052, -6.291, -10.242, -14.049, -17.82, -21.726, -25.884, -30.546, -36, -42.768]),
        'ELEVATION_delta_lower':np.ndarray(16),
        'ELEVATION_delta_upper':np.ndarray(16),
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}
dictionary = EAS2_bin_dict_lessold

dictionary['ELEVATION_delta_upper'][0] = 45-dictionary['ELEVATION'][0]
dictionary['ELEVATION_delta_lower'][0] = dictionary['ELEVATION_delta_upper'][0]
for i in range(1, len(dictionary['ELEVATION'])):
        dictionary['ELEVATION_delta_upper'][i] = dictionary['ELEVATION'][i-1] - dictionary['ELEVATION_delta_upper'][i-1] - dictionary['ELEVATION'][i]
        dictionary['ELEVATION_delta_lower'][i] = dictionary['ELEVATION_delta_upper'][i]

dictionary['ELEVATION_lower_bound'] = dictionary['ELEVATION'] - dictionary['ELEVATION_delta_lower'] # subtract elevation bin lower deltas from elevation bin centers to get the lower bounds
dictionary['AZIMUTH_lower_bound'] = dictionary['AZIMUTH'] - dictionary['AZIMUTH_delta_lower']
dictionary['ELEVATION_upper_bound'] = dictionary['ELEVATION'] + dictionary['ELEVATION_delta_upper'] # add elevation bin upper deltas to elevation bin centers to get the upper bounds
dictionary['AZIMUTH_upper_bound'] = dictionary['AZIMUTH'] + dictionary['AZIMUTH_delta_upper']
dictionary['ELEVATION_bin_count'] = len(dictionary['ELEVATION'])
dictionary['AZIMUTH_bin_count'] = len(dictionary['AZIMUTH'])

EAS1_bin_dict_lessold = {'ELEVATION':np.array([38.394, 27.99, 19.818, 13.086, 7.461, 2.655, -1.548, -5.328, -8.856, -12.258, -15.669, -19.242, -23.148, -27.63, -33.039, -39.789]),
        'ELEVATION_delta_lower':np.ndarray(16),
        'ELEVATION_delta_upper':np.ndarray(16),        
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}
dictionary = EAS1_bin_dict_lessold

dictionary['ELEVATION_delta_upper'][0] = 45-dictionary['ELEVATION'][0]
dictionary['ELEVATION_delta_lower'][0] = dictionary['ELEVATION_delta_upper'][0]
for i in range(1, len(dictionary['ELEVATION'])):
        dictionary['ELEVATION_delta_upper'][i] = dictionary['ELEVATION'][i-1] - dictionary['ELEVATION_delta_upper'][i-1] - dictionary['ELEVATION'][i]
        dictionary['ELEVATION_delta_lower'][i] = dictionary['ELEVATION_delta_upper'][i]

dictionary['ELEVATION_lower_bound'] = dictionary['ELEVATION'] - dictionary['ELEVATION_delta_lower'] # subtract elevation bin lower deltas from elevation bin centers to get the lower bounds
dictionary['AZIMUTH_lower_bound'] = dictionary['AZIMUTH'] - dictionary['AZIMUTH_delta_lower']
dictionary['ELEVATION_upper_bound'] = dictionary['ELEVATION'] + dictionary['ELEVATION_delta_upper'] # add elevation bin upper deltas to elevation bin centers to get the upper bounds
dictionary['AZIMUTH_upper_bound'] = dictionary['AZIMUTH'] + dictionary['AZIMUTH_delta_upper']
dictionary['ELEVATION_bin_count'] = len(dictionary['ELEVATION'])
dictionary['AZIMUTH_bin_count'] = len(dictionary['AZIMUTH'])

lessold_bin_dictionary = (EAS1_bin_dict_lessold, EAS2_bin_dict_lessold)



EAS2_bin_dict_lastonboard = {'ELEVATION':np.ndarray(16),
        'ELEVATION_delta_lower':np.ndarray(16),
        'ELEVATION_delta_upper':np.ndarray(16),
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}
dictionary = EAS2_bin_dict_lastonboard

dictionary['ELEVATION_lower_bound'] = [39.0, 32.55, 26.55, 20.85, 15.35, 9.96667, 4.66667, -0.583333, -5.83333, -11.1, -16.4167, -21.8333, -27.45, -33.3667, -39.7667, -45.0]
dictionary['AZIMUTH_lower_bound'] = dictionary['AZIMUTH'] - dictionary['AZIMUTH_delta_lower']
dictionary['ELEVATION_upper_bound'] = [45.0, 39.0, 32.55, 26.55, 20.85, 15.35, 9.96667, 4.66667, -0.583333, -5.83333, -11.1, -16.4167, -21.8333, -27.45, -33.3667, -39.7667]
dictionary['AZIMUTH_upper_bound'] = dictionary['AZIMUTH'] + dictionary['AZIMUTH_delta_upper']
dictionary['ELEVATION_bin_count'] = len(dictionary['ELEVATION'])
dictionary['AZIMUTH_bin_count'] = len(dictionary['AZIMUTH'])

for i in range(len(dictionary['ELEVATION_delta_lower'])):
        delta = (dictionary['ELEVATION_upper_bound'][i] - dictionary['ELEVATION_lower_bound'][i])/2
        dictionary['ELEVATION_delta_lower'][i] = delta
        dictionary['ELEVATION_delta_upper'][i] = delta
        dictionary['ELEVATION'][i] = dictionary['ELEVATION_upper_bound'][i] + delta

EAS1_bin_dict_lastonboard = {'ELEVATION':np.ndarray(16),
        'ELEVATION_delta_lower':np.ndarray(16),
        'ELEVATION_delta_upper':np.ndarray(16),
        'AZIMUTH':np.array([5.625,  16.875,  28.125,  39.375,  50.625,  61.875,  73.125,  84.375,  95.625, 106.875, 118.125, 129.375, 140.625, 151.875, 163.125, 174.375, 185.625, 196.875, 208.125, 219.375, 230.625, 241.875, 253.125, 264.375, 275.625, 286.875, 298.125, 309.375, 320.625, 331.875, 343.125, 354.375]),
        'AZIMUTH_delta_lower':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625]),
        'AZIMUTH_delta_upper':np.array([5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625, 5.625])}
dictionary = EAS1_bin_dict_lastonboard

dictionary['ELEVATION_lower_bound'] = [39.0, 32.55, 26.55, 20.85, 15.35, 9.96667, 4.66667, -0.583333, -5.83333, -11.1, -16.4167, -21.8333, -27.45, -33.3667, -39.7667, -45.0]
dictionary['AZIMUTH_lower_bound'] = dictionary['AZIMUTH'] - dictionary['AZIMUTH_delta_lower']
dictionary['ELEVATION_upper_bound'] = [45.0, 39.0, 32.55, 26.55, 20.85, 15.35, 9.96667, 4.66667, -0.583333, -5.83333, -11.1, -16.4167, -21.8333, -27.45, -33.3667, -39.7667]
dictionary['AZIMUTH_upper_bound'] = dictionary['AZIMUTH'] + dictionary['AZIMUTH_delta_upper']
dictionary['ELEVATION_bin_count'] = len(dictionary['ELEVATION'])
dictionary['AZIMUTH_bin_count'] = len(dictionary['AZIMUTH'])

for i in range(len(dictionary['ELEVATION_delta_lower'])):
        delta = (dictionary['ELEVATION_upper_bound'][i] - dictionary['ELEVATION_lower_bound'][i])/2
        dictionary['ELEVATION_delta_lower'][i] = delta
        dictionary['ELEVATION_delta_upper'][i] = delta
        dictionary['ELEVATION'][i] = dictionary['ELEVATION_upper_bound'][i] + delta

lastonboard_bin_dictionary = (EAS1_bin_dict_lastonboard, EAS2_bin_dict_lastonboard)



gennaroEASXtop = [[45.00, 39.00, 32.55, 26.55, 20.85, 15.35, 9.97, 4.67, -0.58, -5.83, -11.10, -16.42, -21.83, -27.45, -33.37, -39.77], [45.00, 39.00, 32.55, 26.55, 20.85, 15.35, 9.97, 4.67, -0.58, -5.83, -11.10, -16.42, -21.83, -27.45, -33.37, -39.77]]
gennaroEASXbot = [[39.00, 32.55, 26.55, 20.85, 15.35, 9.97, 4.67, -0.58, -5.83, -11.10, -16.42, -21.83, -27.45, -33.37, -39.77, -45.00], [39.00, 32.55, 26.55, 20.85, 15.35, 9.97, 4.67, -0.58, -5.83, -11.10, -16.42, -21.83, -27.45, -33.37, -39.77, -45.00]]
gennaro_bin_dictionary = fx.getBinDictsFromBounds(gennaroEASXtop,gennaroEASXbot)



missedmissionstartEASXtop = [list(map(float,'44.37 33.53 24.25 16.23 9.27 3.01 -2.39 -7.16 -11.22 -14.82 -18.23 -21.53 -24.81 -28.50 -32.93 -38.48'.split(" "))), list(map(float,'43.95 32.26 23.00 14.77 7.57 1.59 -3.41 -7.69 -11.32 -14.62 -17.71 -20.57 -23.78 -27.54 -32.26 -38.40'.split(" ")))]
missedmissionstartEASXbot = [list(map(float,'34.69 25.25 17.09 10.03 3.69 -1.80 -6.65 -10.78 -14.42 -17.85 -21.16 -24.42 -28.04 -32.36 -37.77 -44.60'.split(" "))), list(map(float, '34.14 24.56 16.15 8.76 2.58 -2.57 -6.99 -10.70 -14.05 -17.18 -20.04 -23.18 -26.81 -31.32 -37.23 -44.36'.split(" ")))]
missedmissionstart_bin_dictionary = fx.getBinDictsFromBounds(missedmissionstartEASXtop,missedmissionstartEASXbot)



aug2020EASXtop = [list(map(float,'45.00 33.68 24.66 17.16 10.80 5.32 0.50 -3.82 -7.81 -11.59 -15.27 -18.99 -22.89 -27.12 -31.94 -37.70'.split(" "))), list(map(float,'45.00 32.88 23.62 16.10 9.87 4.62 0.08 -3.94 -7.62 -11.12 -14.56 -18.08 -21.86 -26.08 -31.06 -37.20'.split(" ")))]
aug2020EASXbot = [list(map(float,'33.68 24.66 17.16 10.80 5.32 0.50 -3.82 -7.82 -11.59 -15.27 -18.99 -22.89 -27.11 -31.94 -37.70 -45.02'.split(" "))), list(map(float, '32.88 23.62 16.10 9.88 4.63 0.08 -3.94 -7.62 -11.12 -14.56 -18.08 -21.86 -26.08 -31.06 -37.20 -45.00'.split(" ")))]
aug2020_bin_dictionary = fx.getBinDictsFromBounds(aug2020EASXtop,aug2020EASXbot)

#print(EAS1_bin_dict['ELEVATION_lower_bound'])
#print(EAS1_bin_dict['ELEVATION'])
#print(EAS1_bin_dict['ELEVATION_upper_bound'])
#print(EAS1_bin_dict['AZIMUTH_lower_bound'])
#print(EAS1_bin_dict['AZIMUTH'])
#print(EAS1_bin_dict['AZIMUTH_upper_bound'])
#print('HOOOLD IIIIT')