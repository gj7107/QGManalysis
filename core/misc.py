import os
import scipy.io, scipy
import numpy as np
from skimage import restoration
import scipy.optimize
import matplotlib.pyplot as plt

def get_all_files(folder_path, extension) :
    return [f for f in os.listdir(folder_path) if f.endswith(extension) and os.path.isfile(os.path.join(folder_path, f))]

def get_newest_file(folder_path, extension):
    with os.scandir(folder_path) as entries:
        files = ((entry.path, entry.stat().st_ctime) for entry in entries if entry.is_file() and entry.path.endswith(extension))
        newest_file = max(files, key=lambda x: x[1], default=(None, None))[0]
    
    return newest_file


def loadLatticeData(path) :
    mat = scipy.io.loadmat(path)
    PSF = mat['PSF']
    lattice_a = mat['lattice_a']
    return PSF, lattice_a

def GenerateAnalysisParameters(ROI, lattice_a, PSF) :
    AnalysisParameters = {
        'px': [0.0004076,-0.01447,0.1764,-0.163],
        'py': [-0.0001152,0.002677,-0.02554,0.02305],
        'imgsize':[800,800],
        'n':2,
        'thlevels':np.ones((1,5))*0.5,   
        #'overwrite':True,
        'imgth':0.6,
        'ndiv':15,
        'BeamIntensityCalibration':True,
        'ROI':ROI,
        'lattice_a':lattice_a, 
        'PSF':PSF,
        'LockPoint':{'x':0,'y':0}
    }
    return AnalysisParameters

def exist_analysisfile(directory, fname) :
    fname1 = os.path.splitext(fname)[0] + "_Lat.mat"
    fname2 = fname + "_Lat.mat"
    return os.path.isfile(os.path.join(directory, fname1)) or os.path.isfile(os.path.join(directory, fname2))
