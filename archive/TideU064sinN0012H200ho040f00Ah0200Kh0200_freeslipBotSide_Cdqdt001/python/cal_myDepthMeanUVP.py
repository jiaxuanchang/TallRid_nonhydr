from xmitgcm import open_mdsdataset
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib import cm
import pandas as pd
import numpy as np
import math
from scipy import integrate
import xarray as xr

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds1 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energyvars','statevars','statevars2d'])
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])

t = 100

ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )

time1=ds1.coords['time']
time2=ds2.coords['time']
xc=ds1.coords['XC']
xg=ds1.coords['XG']
yc=ds1.coords['YC']
yg=ds1.coords['YG']
z=ds1.coords['Z']

PS = 0.5*(ds1['PHIHYD'].roll(YC=1).values+ds1['PHIHYD'].values) 
PW = 0.5*(ds1['PHIHYD'].roll(XC=1).values+ds1['PHIHYD'].values)
#print(np.shape(PS))
#print('PHIHYD=' + str(ds1['PHIHYD'].isel(YC=0,time=100,Z=2).values))
#print('PS='+ str(PS[100,2,0,:]))
#print('PW='+ str(PW[100,:,0,1]))
#print(ds1['PHIHYD'].roll(XC=1)+ds1['PHIHYD'])
#print(ds1['PHIHYD'].roll(XC=1).values[100,2,0,:])
#print(ds1['PHIHYD'].values[100,2,0,:])

#when use interp-> can't with .values???  can't work
#US=xr.DataArray(ds1['UVEL'].interp(XG=xc,YC=yg,kwargs={'fill_value':'extrapolate'}).data,coords=[time1,z,yg,xc],dims=['time','Z','YG','XC'])
#print(US.values)
#VW=(ds1['VVEL'].interp(XC=xg,YG=yc, kwargs={'fill_value': 'None'})).values()

# this US.VW only correct in 2d cases
US=0.5*(ds1['UVEL'].roll(XG=-1).values+ds1['UVEL'].values)
VW=0.5*(ds1['VVEL'].roll(XC=1).values+ds1['VVEL'].values) #VW(when X=0, incorrect

#print('U:' + str(ds1['UVEL'].isel(YC=0,time=100,Z=2).values))
#print('US:' +str(US[100,2,0,:]))
#print('V:' +str(ds1['VVEL'].isel(YG=0,time=100,Z=2).values))
#print('VW:' + str(VW[100,2,0,:]))

#depth mean pressure
P0S=np.sum(PS*(ds1['drF']*ds1['hFacS']).values,axis=1)
P0W=np.sum(PW*(ds1['drF']*ds1['hFacW']).values,axis=1)
#print('P0W=' +str(P0W[100,0,:]))
#print(np.shape(P0W))

#depth mean velocity
U0W=((ds1['UVEL']*ds1['drF']*ds1['hFacW']).sum('Z')).values
U0S= np.sum(US*(ds1['drF']*ds1['hFacS']).values,axis=1)
V0W= np.sum(VW*(ds1['drF']*ds1['hFacW']).values,axis=1)
V0S=((ds1['VVEL']*ds1['drF']*ds1['hFacS']).sum('Z')).values
#print(np.shape(U0W))
#print(np.shape(U0S))

ZW=((ds1['drF']*ds1['hFacW']).sum('Z')).values
ZS=((ds1['drF']*ds1['hFacS']).sum('Z')).values
#print(np.shape(ZW))
#print('ZW=' +str(ZW[0,:]))
#print(ds1)

for i in range(400):
    if ZW[0,i]!=0:
        P0W[:,0,i]=P0W[:,0,i]/ZW[0,i]
        U0W[:,0,i]=U0W[:,0,i]/ZW[0,i]
        V0W[:,0,i]=V0W[:,0,i]/ZW[0,i]
    else:
        P0W[:,0,i]=0
        U0W[:,0,i]=0
        V0W[:,0,i]=0

print(P0W[100,0,:])

for i in range(400):
    if ZS[0,i]!=0:
        P0S[:,0,i]=P0S[:,0,i]/ZS[0,i]
        U0S[:,0,i]=U0S[:,0,i]/ZS[0,i]
        V0S[:,0,i]=V0S[:,0,i]/ZS[0,i]
    else:
        P0S[:,0,i]=0
        U0S[:,0,i]=0
        V0S[:,0,i]=0
print(P0S[100,0,:])
print(U0W[100,0,:])

p_bt=(ds1['PHIHYD']*ds1['drF']*ds1['hFacC']).isel(time=t,YC=0).sum('Z')
dc=(ds1['drF']*ds1['hFacC']).isel(YC=0).sum('Z')
dm_p=p_bt/dc
u_bt=(ds1['UVEL']*ds1['drF']*ds1['hFacW']).isel(time=t,YC=0).sum('Z')
dw=(ds1['drF']*ds1['hFacW']).isel(YC=0).sum('Z')
dm_u=u_bt/dw
v_bt=(ds1['VVEL']*ds1['drF']*ds1['hFacS']).isel(time=t,YG=0).sum('Z')
ds=(ds1['drF']*ds1['hFacS']).isel(YG=0).sum('Z')
dm_v=v_bt/ds

if 1:
    fig= plt.figure(figsize=(9,10))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 3.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ds1['PHIHYD'].isel(time=t,YC=0).plot(ax=ax1,cmap='RdBu_r',cbar_kwargs={"aspect":40})
    ax1.set_ylabel('Z')
    ax1.set_xlabel('X')
    dm_p.plot(ax=ax2)
    ax2.set_xlabel('X')
    ax2.set_title('depth-mean P')
    cset1=ax3.pcolormesh(xc,time1,np.squeeze(P0S),cmap='RdBu_r')
    plt.colorbar(cset1,ax=ax3,aspect=40)
    ax3.set_ylabel('time')
    ax3.set_xlabel('X')
    ax3.plot(xc,np.ones(400)*time1[t].values,'--k')
    ax3.set_title('P0S')
    cset2=ax4.pcolormesh(xg,time1,np.squeeze(P0W),cmap='RdBu_r')
    plt.colorbar(cset2,ax=ax4,aspect=40)
    ax4.set_xlabel('X')
    ax4.set_title('P0W')
    ax4.plot(xc,np.ones(400)*time1[t].values,'--k')
    plt.savefig('./figs/depthmeanPressure.png')

if 1:
    plt.clf()
    fig= plt.figure(figsize=(9,10))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 3.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ds1['UVEL'].isel(time=t,YC=0).plot(ax=ax1,cmap='RdBu_r',cbar_kwargs={"aspect":40})
    ax1.set_ylabel('Z')
    ax1.set_xlabel('X')
    dm_u.plot(ax=ax2)
    ax2.set_xlabel('X')
    ax2.set_title('depth-mean U')
    cset1=ax3.pcolormesh(xc,time1,np.squeeze(U0S),cmap='RdBu_r')
    plt.colorbar(cset1,ax=ax3,aspect=40)
    ax3.set_ylabel('time') 
    ax3.set_xlabel('X')
    ax3.set_title('U0S')
    ax3.plot(xc,np.ones(400)*time1[t].values,'--k')  
    cset2=ax4.pcolormesh(xg,time1,np.squeeze(U0W),cmap='RdBu_r')
    ax4.set_xlabel('X') 
    ax4.set_title('U0W')
    ax4.plot(xc,np.ones(400)*time1[t].values,'--k')  
    plt.savefig('./figs/depthmeanUVEL.png')

if 1:
    plt.clf()
    fig= plt.figure(figsize=(9,10))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 3.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ds1['VVEL'].isel(time=t,YG=0).plot(ax=ax1,cmap='RdBu_r',cbar_kwargs={"aspect":40})
    ax1.set_ylabel('Z')
    ax1.set_xlabel('X')
    dm_v.plot(ax=ax2)
    ax2.set_xlabel('X')
    ax2.set_title('depth-mean V')
    cset1=ax3.pcolormesh(xc,time1,np.squeeze(V0S),cmap='RdBu_r')
    plt.colorbar(cset1,ax=ax3,aspect=40)
    ax3.set_ylabel('time')
    ax3.set_xlabel('X')
    ax3.set_title('V0S')
    ax3.plot(xc,np.ones(400)*time1[t].values,'--k')  
    cset2=ax4.pcolormesh(xg,time1,np.squeeze(V0W),cmap='RdBu_r')
    plt.colorbar(cset2,ax=ax4,aspect=40)
    ax4.set_xlabel('X')
    ax4.set_title('V0W')
    ax4.plot(xc,np.ones(400)*time1[t].values,'--k')  
    plt.savefig('./figs/depthmeanVVEL.png')




