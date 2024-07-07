# This code generates a high quality gif given a directory where all your images (and only your images) are saved, and saves it in that same directory. The gif must be deleted externally before running this code again.

import numpy as np
import os
import imageio.v3 as iio

directory = 'animations\\20230612h17m29s19μ000000_to_20230612h17m29s29μ000000_justEAS_fullcontours'
gif_path = directory + "\\20230612h17m29s19μ000000_justEAS_fullcontours_gif" + ".gif"
frameslist = os.listdir(directory)
#print(frameslist)

frames = np.stack([iio.imread(directory + f"\{framename}") for framename in frameslist], axis=0)

iio.imwrite(gif_path, frames, duration=0.125, loop=0)