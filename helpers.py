# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 21:35:44 2017

@author: Ania
"""

import h5py
import numpy as np
from PIL import Image

def load_hdf5(infile):
  with h5py.File(infile,"r") as f:  #"with" close the file after its nested commands
    return f["image"][()]

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)
