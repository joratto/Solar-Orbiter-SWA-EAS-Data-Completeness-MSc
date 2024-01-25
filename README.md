# Joaquim-MSc
Determining the accuracy and completeness of Burst Mode data collection from the Solar Orbiter SWA electron sensor.

To use spacepy.pycdf, you must first install the NASA CDF C library from https://cdf.gsfc.nasa.gov/html/sw_and_docs.html
then, before importing spacepy.pycdf, you must include: os.environ["CDF_LIB"] = "PATH" where PATH is the location where the CDF C library is installed (e.g. "C:\\Program Files\\CDF_Distribution").
