import sys
print(sys.executable)
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CDF_LIB"] = "C:\\Program Files\\CDF_Distribution"
from spacepy import pycdf

print('Hello Joaquim')

from spacepy import pycdf
import datetime
time = [datetime.datetime(2000, 10, 1, 1, val) for val in range(60)]
import numpy as np

cdf_path = 'data\solo_L2_mag-rtn-normal-1-minute_20231031_V01.cdf'
cdf = pycdf.CDF(cdf_path)
print(len(cdf['VECTOR_TIME_RESOLUTION']))
print(cdf['VECTOR_TIME_RESOLUTION'][0])
print(cdf['EPOCH'][1])
B_rtn = np.array(cdf['B_RTN'])

Bx, By, Bz = B_rtn.T

t = np.array(cdf['EPOCH'])

plt.subplot(111)

plt.plot(t,Bx)
plt.plot(t,By)
plt.plot(t,Bz)
plt.ylabel('B (nT)')
plt.xlabel('Date & Time (MM-DD HH)')
plt.legend(["Radial", "Tangent", "Normal"])
plt.grid()
plt.title(cdf_path[5:])

plt.show()

""" data = np.random.random_sample(len(time))
cdf = pycdf.CDF('test3.cdf', '')
cdf['Epoch'] = time
cdf['data'] = data
cdf.attrs['Author'] = 'John Doe'
cdf.attrs['CreateDate'] = datetime.datetime.now()
cdf['data'].attrs['units'] = 'MeV'


import datetime
# make a dataset every minute for a hour
time = [datetime.datetime(2000, 10, 1, 1, val) for val in range(60)]

# put time into CDF variable Epoch
cdf['Epoch'] = time
# and the same with data (the smallest data type that fits the data is used by default)
cdf['data'] = data

# add some attributes to the CDF and the data
cdf.attrs['Author'] = 'John Doe'
cdf.attrs['CreateDate'] = datetime.datetime.now()
cdf['data'].attrs['units'] = 'MeV'

cdf.close()
 """