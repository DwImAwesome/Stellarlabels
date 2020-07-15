import time
import h5py
import numpy as np

import torch
from numpy import genfromtxt # function to read in the wavelengths

def get_data(datastr):
    with h5py.File("../data/"+datastr, "r") as hf :
        spectrum = hf["spectrum"][:]
        teff = hf["TEFF"][:]
        logg = hf["LOGG"][:]
        Z = hf["Z"][:]
        vsini = hf["VSINI"][:]
    y = np.vstack([teff, logg, Z, vsini])
    y = np.transpose(y)
    return y, spectrum


def load_data(channelwise):
    start = time.time()
    # read in wavelengths
    wavelengths = genfromtxt("../data/wl_solution.txt", delimiter=',')

    print("load data")
    
    ## load data from file
    y, spectrum = get_data("noisefree_modelgrid.h5")

    if channelwise:
        # Have spectrum as [n, 8, 4096]
        #spectrum = torch.from_numpy(spectrum).float()
        spectrum = torch.from_numpy(spectrum).float().unsqueeze(1)
    else:
        # Have spectrum as [n, 1, 4096*8]
        spectrum = torch.from_numpy(spectrum).view(-1,4096*8).unsqueeze(1).float()
        # Remove zero_padding
        spectrum = spectrum[:,:,3000:(32768-3000)]

    print("Loaded in", time.time()-start, "seconds")

    print('Final shape of data set:', spectrum.shape)
    y = torch.from_numpy(y).float()
    return spectrum, y
