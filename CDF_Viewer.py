import os
os.environ["CDF_LIB"] = "C:\\Program Files\\CDF_Distribution"
from spacepy import pycdf

cdf_filename = 'solo_L1_swa-eas-padc_20230530T172732-20230530T173231_V01.cdf'
cdf_eas = pycdf.CDF('data\\' + cdf_filename)

print(cdf_eas)