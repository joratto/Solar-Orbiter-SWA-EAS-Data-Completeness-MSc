import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["CDF_LIB"] = "C:\\Program Files\\CDF_Distribution"
from spacepy import pycdf
import datetime

cdf_filename_eas = 'solo_L1_swa-eas-padc_20230831T172734-20230831T173233_V01.cdf'
cdf_eas = pycdf.CDF('data\\' + cdf_filename_eas)
#print(cdf_eas)
#print(cdf_eas['SWA_EAS_MagDataUsed'][:])
time_eas = np.array(cdf_eas['EPOCH'])
t0 = time_eas[0] # start of timeframe
tf = time_eas[-1] # end of timeframe
print('EAS time series from {} to {}'.format(t0, tf))
#for vector in cdf_eas['SWA_EAS_MagDataUsed']:
#    print(np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2))

cdf_filename_mag = 'solo_L2_mag-srf-normal_20230831_V01.cdf'
cdf_mag = pycdf.CDF('data\\' + cdf_filename_mag)
#print(cdf_mag)
time_mag = np.array(cdf_mag['EPOCH'][500000:])

length = len(time_mag)

# The following code block finds where to crop time_mag and
#B_mag to match time_eas and B_eas (can be more efficient).
print('\n')
time_mag_index_t0 = 0
while time_mag[time_mag_index_t0] < t0: #datetime.datetime(1970,1,1):#t0:
    time_mag_index_t0 += 1
    print('\rt0 MAG index = {}/{}'.format(time_mag_index_t0,length), end='')
    continue
print('\n')
time_mag_index_tf = time_mag_index_t0
while time_mag[time_mag_index_tf] < tf: #datetime.datetime(1970,1,1):#
    time_mag_index_tf += 1
    print('\rtf MAG index = {}/{}'.format(time_mag_index_tf,length), end='')
    continue
print('\n')

time_mag = time_mag[time_mag_index_t0:time_mag_index_tf]

B_mag = cdf_mag['B_SRF'][time_mag_index_t0:time_mag_index_tf] # reported MAG B, cropped to match EAS timeframe
B_eas = cdf_eas['SWA_EAS_MagDataUsed'] # onboard EAS B

# normalise B_mag: 
B_mag_norm = np.array([])
for i in range(len(B_mag)):
    vector = B_mag[i]
    B_mag[i] = vector/np.linalg.norm(vector)

Bx_mag, By_mag, Bz_mag = np.array(B_mag).T
Bx_eas, By_eas, Bz_eas = np.array(B_eas).T

sns.set_theme(style='ticks')
fig, axs = plt.subplots(3)

axs[0].plot(time_mag,Bx_mag)
axs[0].plot(time_eas,Bx_eas)
axs[0].set_ylabel('Bx (normalised)')
#axs[0].set_xlabel('Date & Time (dd hh:mm)')
axs[0].legend(["Bx MAG", "Bx EAS"])
axs[0].grid()
axs[0].set_title('{}    &    {}'.format(cdf_filename_eas, cdf_filename_mag))

axs[1].plot(time_mag,By_mag)
axs[1].plot(time_eas,By_eas)
axs[1].set_ylabel('By (normalised)')
axs[1].legend(["By MAG", "By EAS"])
axs[1].grid()

axs[2].plot(time_mag,Bz_mag)
axs[2].plot(time_eas,Bz_eas)
axs[2].set_ylabel('Bz (normalised)')
axs[2].legend(["Bz MAG", "Bz EAS"])
axs[2].grid()
axs[2].set_xlabel('Date & Time (dd hh:mm)')

# quick calculation of the angle difference between B vectors
print('{} degrees'.format(180*(np.arctan(Bz_mag[0]/Bx_mag[0])/90/np.pi - np.arctan(Bz_eas[0]/Bx_eas[0])/np.pi)))
print('{} degrees'.format(180*(np.arctan(By_mag[0]/Bx_mag[0])/90/np.pi - np.arctan(By_eas[0]/Bx_eas[0])/90/np.pi)))

plt.show()

