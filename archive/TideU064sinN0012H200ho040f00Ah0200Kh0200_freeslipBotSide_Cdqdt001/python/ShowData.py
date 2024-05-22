from xmitgcm import open_mdsdataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import string

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

#ds1 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energyvars','statevars','statevars2d'])
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])
yc=ds2.coords['YC']

figs={}
if 0:
    for t in range(150,200,5):
        figs[t]=plt.figure()
        U0_10=ds['UVEL'].isel(time=t,YC=0)
        U0_10.plot()

print(ds2)
if 1:
    for iy in range(len(yc)):
        f, ax = plt.subplots(1, 6, figsize=(22,9) , sharey=True)
        ds2['SDIAG5'].isel(YC=iy).plot(ax=ax[0],y='time',cbar_kwargs={"label": "", "aspect": 40})
        ax[0].set_title('Ebc at yc=%05d' %yc[iy] )
        ds2['SDIAG6'].isel(YC=iy).plot(ax=ax[1],y='time',cbar_kwargs={"label": "", "aspect": 40})
        ax[1].set_title('uPbc at yc=%05d' %yc[iy])
        ds2['SDIAG7'].isel(YC=iy).plot(ax=ax[2],y='time',cbar_kwargs={"label": "", "aspect": 40})
        ax[2].set_title('vPbc')
        ds2['SDIAG8'].isel(YG=iy).plot(ax=ax[3],y='time',cbar_kwargs={"label": "", "aspect": 40})
        ax[3].set_title('uEbc')
        ds2['SDIAG9'].isel(YC=iy).plot(ax=ax[4],y='time',cbar_kwargs={"label": "", "aspect": 40})
        ax[4].set_title('vEbc')
        ds2['SDIAG10'].isel(YG=iy).plot(ax=ax[5],y='time',cbar_kwargs={"label": "", "aspect": 40})
        ax[5].set_title('Conv')
        

        for n, axs in enumerate(ax):
            axs.text(-0.1, 1, string.ascii_lowercase[n], transform=axs.transAxes, 
                                        size=20, weight='bold')
        plt.tight_layout()
        plt.savefig('./figs/energy_diagnostics_iy%d.png' %iy)

#plt.show()
