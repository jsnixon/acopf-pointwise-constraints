import h5py

with h5py.File('case30_ieee.h5', 'r') as f:
    for key in f.keys():
        print(f"{key}: {f[key].shape}")