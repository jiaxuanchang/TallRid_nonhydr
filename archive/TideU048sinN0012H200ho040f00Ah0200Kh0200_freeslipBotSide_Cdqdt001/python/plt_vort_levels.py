from xmitgcm import open_mdsdataset
import xgcm
import psyplot.project as psy
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import math
import xarray as xr
import string

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds1 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energyvars','statevars','statevars2d'])
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])
grid = xgcm.Grid(ds1, periodic=False)
print(grid)

t = 0
h0 = 140.
H = 200.
f0 = 1.e-4
g = 9.8

om=2*np.pi/12.4/3600
alpha = 2e-4
nz = 200
dz=H/nz 

tR_fname="../indata/TRef.bin"
tRef = np.fromfile(tR_fname)
refSalt=35.
refTemp=tRef[0]
print('refTemp='+ str(refTemp))
print('tRef='+ str(tRef))

xmin = -5
xmax = 5
vmax = 8e-3
vmin = -8e-3
numcolt=21
numcolv=21
                                        
ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )

cmap = cm.RdBu_r
levels = np.linspace(vmin, vmax, num=numcolv)
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#pcolor_opts1 = {'cmap':cm.get_cmap(cmap, len(levels) - 1), 'norm': norm}

print(ds1)

xg=ds1.coords["XG"]
xc=ds1.coords["XC"]
yg=ds1.coords["YG"]
yc=ds1.coords["YC"]
time=ds1.coords["time"].values/np.timedelta64(1, 's')/3600
XC0 = (ds1.XC/1000.-ds1.XC.mean()/1000.).values
YC0 = (ds1.YC/1000.-ds1.YC.mean()/1000.).values
XG0 = (ds1.XG/1000.-ds1.XC.mean()/1000.).values
YG0 = (ds1.YG/1000.-ds1.YC.mean()/1000.).values
Z=ds1.coords["Z"]

windowsize=5

#cal vorticity

U=ds1['UVEL'].isel(time=0)
V=ds1['VVEL'].isel(time=0)
UC=grid.interp(U,'X',boundary='extrapolate')
VC=grid.interp(V,'Y',boundary='extrapolate')
maskZ = grid.interp(ds1.hFacS, 'X', boundary='extend')

zeta = (-grid.diff(U * ds1['dxC'], 'Y', boundary='extend') + grid.diff(V * ds1['dyC'], 'X',boundary='extend'))/ds1['rAz']
zeta = xr.DataArray(zeta,coords=[Z,YG0,XG0],dims=["Z","YG0","XG0"])
maskZ = xr.DataArray(maskZ,coords=[Z,YG0,XG0],dims=["Z","YG0","XG0"])
zeta['XG0'].attrs["units"] = "km"
zeta['YG0'].attrs["units"] = "km"
maskZ['XG0'].attrs["units"] = "km"
maskZ['YG0'].attrs["units"] = "km"

zeta = zeta.sel(XG0=XG0[(XG0 > xmin) & (XG0 < xmax)])
zeta = zeta.isel(Z=slice(0,36,3))
maskZ = maskZ.sel(XG0=XG0[(XG0 > xmin) & (XG0 < xmax)])
maskZ = maskZ.isel(Z=slice(0,36,3))

s=range(0,36,3)
zetachunked=zeta.chunk(chunks=(12,60,400))
# smooth vorticity
sm_zeta = zetachunked.rolling(XG0=windowsize,center=True).mean()
sm_zeta = sm_zeta.rolling(YG0=windowsize,center=True).mean()

# coarse grid for vector
from scipy.interpolate import griddata
print(xc.shape)
xx,yy=np.meshgrid(xc.values,yc.values)
print(np.shape(xx))
points=np.transpose([xx.flatten(),yy.flatten()])
print(np.shape(points))
grid_x, grid_y = np.mgrid[41000:47000:200,0:3000:100]
grid_x0=grid_x/1000.-ds1.XC.values.mean()/1000.
grid_y0=grid_y/1000.-ds1.YC.values.mean()/1000.
print(grid_x0[0,:5])

#current vector
UC=UC.where(ds1['maskC'] !=0,np.nan)
VC=VC.where(ds1['maskC'] !=0,np.nan)
UC=UC.isel(Z=slice(0,36,3))
VC=VC.isel(Z=slice(0,36,3))

# plotting
fig, axes = plt.subplots(figsize=(16,9),nrows=3, ncols=4,sharex='all', sharey='all')
fig.subplots_adjust(top=0.9,left=0.08,right=0.93)
fig.suptitle("t= %2.2f hrs" %time[t], size=16)
patch_list = []
vec_list=[]

for ind, ax, in enumerate(axes.flatten()):
    pm=sm_zeta.where(maskZ).isel(Z=ind).plot(ax=ax,levels=levels,cmap='RdBu_r')
    grid_U = griddata(points,UC.isel(Z=ind).values.flatten(),(grid_x, grid_y))
    grid_V = griddata(points,VC.isel(Z=ind).values.flatten(),(grid_x, grid_y))
    quiv=ax.quiver(grid_x0, grid_y0, grid_U, grid_V, units='x',scale=1 / 0.3, color='black')
    ax.set_title('Z= %2.1f m' %zeta['Z'][ind]) 
    ax.set_xlim(xmin,xmax)
    patch_list.append(pm)
    vec_list.append(quiv)

print(patch_list)

tt: int=0
def updateData(tt):
    time = ds1.time.values[tt]/np.timedelta64(1, 's')/3600
    U=ds1['UVEL'].isel(time=tt)
    V=ds1['VVEL'].isel(time=tt)
    UC=grid.interp(U,'X',boundary='extrapolate')
    VC=grid.interp(V,'Y',boundary='extrapolate')
    
    zeta = (-grid.diff(U * ds1['dxC'], 'Y', boundary='extend') + grid.diff(V * ds1['dyC'], 'X',boundary='extend'))/ds1['rAz']
    zeta = xr.DataArray(zeta,coords=[Z,YG0,XG0],dims=["Z","YG0","XG0"])
    
    zeta = zeta.sel(XG0=XG0[(XG0 > xmin) & (XG0 < xmax)])
    zeta = zeta.isel(Z=slice(0,36,3))
    zetachunked = zeta.chunk(chunks=(12,60,400))

    sm_zeta = zetachunked.rolling(XG0=windowsize,center=True).mean()
    sm_zeta = sm_zeta.rolling(YG0=windowsize,center=True).mean()
    
    UC=UC.where(ds1['maskC'] !=0,np.nan)
    VC=VC.where(ds1['maskC'] !=0,np.nan)
    UC=UC.isel(Z=slice(0,36,3))
    VC=VC.isel(Z=slice(0,36,3))

    print('tt'+str(tt))
    for ind, feature in enumerate(patch_list): 
        print('ind,feature:')
        print(ind, feature)
        feature.set_array(sm_zeta.where(maskZ).isel(Z=ind).values.ravel())
        fig.suptitle('t= %2.3fhr' % time)
    
    for ind, feature in enumerate(vec_list):
        grid_U = griddata(points,UC.isel(Z=ind).values.flatten(),(grid_x, grid_y))
        grid_V = griddata(points,VC.isel(Z=ind).values.flatten(),(grid_x, grid_y))
        feature.set_UVC(grid_U,grid_V)


simulation = animation.FuncAnimation(fig, updateData, blit=False, frames= ttlen-1 , interval=40, repeat=False)

simulation.save(filename='sm_w5_vort_vect_xy_x3_unitwidthscl3.mp4', fps=2)


