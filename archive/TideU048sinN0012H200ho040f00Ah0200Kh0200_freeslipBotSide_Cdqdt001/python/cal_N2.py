from xmitgcm import open_mdsdataset
import os
import matplotlib.pyplot as plt
import pandas as pd

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds = open_mdsdataset(data_dir, geometry='cartesian', endian='<')
rhoNil=999.8
alpha=2e-4

T=ds['THETA'].isel(XG=0,YC=0)
rho=rhoNil*(1-alpha*(T-Tref))
.plot()


plt.show()
