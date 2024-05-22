from xmitgcm import open_mdsdataset
import xgcm
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import xarray as xr
import string

import                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       warnings
warnings.filterwarnings("ignore")

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

#dst = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['statevars'],iters=range(120*1860,170*1860,1860),levels={'UVEL':[1]})
ds1 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['statevars'],iters=range(120*1860,170*1860,1860))
ds1=ds1.chunk(chunks={"time":30,"YC":120,"YG":120,"XG":360,"XC":360})
print(ds1)

ds = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['statevars'])
sur_bund_slice = {'XC': slice(0,2), 'XG': slice(0,2)
        ,'YC': slice(30,32), 'YG': slice(30,32)
        ,'Z': slice(0,2),'Zl': slice(0,2),'Zp1': slice(0,2),'Zu': slice(0,2)}
sur_bund = ds.isel(**sur_bund_slice)
sur_bund.to_netcdf('./tile_data/surf_bundry.nc')
del ds, sur_bund
sur_bund = open_dataset('./tile_data/surf_bundry.nc')

print(sur_bund)
stop


#ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])
Grid = xgcm.Grid(ds1, periodic=False)


tR_fname="../indata/TRef.bin"
tRef = np.fromfile(tR_fname)
refSalt=35.
refTemp=tRef[0]
print('refTemp='+ str(refTemp))
print('tRef='+ str(tRef))
u0=0.08

xmin = 28
xmax = 52
vmax = 1.0
vmin = -1.0
wmax = 0.2
wmin = -0.2
tmax = math.floor(np.amax(tRef))
tmin = math.floor(np.amin(tRef))
numcolt=15
numcolv=21
                                        
ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )
print(tmin)


xg=ds1.coords["XG"]
xc=ds1.coords["XC"]
yg=ds1.coords["YG"]
yc=ds1.coords["YC"]
z=ds1.coords["Z"]
time=ds1.coords['time'].values/np.timedelta64(1, 's')/3600
print(ds1)
print(yc)
print(xc[725:756].values)
print(list(np.around(xc[725:756].values,0)).index(40762))
print(xc[-2:])
print(ds1.XC.mean())


ds1=ds1.assign_coords(XC0 = (ds1.XC/1000.-ds1.XC.mean()/1000.))
ds1=ds1.assign_coords(XG0 = (ds1.XG-ds1.XC.mean())/1000.)
ds1=ds1.assign_coords(YC0 = (ds1.YC/1000.-ds1.YC.mean()/1000.))
ds1=ds1.assign_coords(YG0 = (ds1.YG-ds1.YC.mean())/1000.)
yc0 = (ds1.YC/1000.-ds1.YC.mean()/1000.)
xc0 = (ds1.XC/1000.-ds1.XC.mean()/1000.)

fig = plt.figure(figsize=(12,12))
grid = plt.GridSpec(6, 6, hspace=0.8, height_ratios=[1.2,1,1, 1,1,1], width_ratios=[1,1,1,1,1,0.2])

# selece plotting slices
sx=[38612,39412,40012,40612,41412] 
sy=[612.5,1212.5,1512.5,1812.5,2412.5]
sz=[-2.5,-17.5,-32.5,-47.5,-62.5]
iy=[]
iz=[]
ix=[]
for i in range(len(sy)):
    print('iy')
    if sy[i] in yc:
        iy.append(list(yc.data).index(sy[i]))
        print(iy)

for i in range(len(sx)):
    print('ix')
    if sx[i] in np.around(xc.values):
        print(sx[i])
        ix.append(list(np.around(xc.values)).index(sx[i]))
        print(ix)

for i in range(len(sz)):
    print('iz')
    if sz[i] in z:
        iz.append(list(z.data).index(sz[i]))
        print(iz)

print(ds1['UVEL'].isel(YC=20, XG=0, Z=0).shape)
print(time.shape)


from scipy.interpolate import griddata
print(xc.shape)
xx,yy=np.meshgrid(xc.values,yc.values)
print(np.shape(xx))
points=np.transpose([xx.flatten(),yy.flatten()])
print(np.shape(points))
grid_x, grid_y = np.mgrid[37000:43000:200,0:3000:100]
grid_x0=grid_x/1000.-ds1.XC.values.mean()/1000.
grid_y0=grid_y/1000.-ds1.YC.values.mean()/1000.
print(grid_x0[0,:5])

#utm=ds1.UVEL.isel(YC=20,XG=0,Z=0).values

for t in range(ttlen):
    #fig.clf()
    ax1=fig.add_subplot(grid[0,:-1])

    #fig1 forcing
    ax1.plot(time,np.zeros(ttlen),color='gray',linestyle='--')
    ax1.plot(time,(sur_bund.UVEL.astype(np.uint8)).isel(Z=0,YC=0,XG=0),label='U(t,0,0,0)')
    ax1.set_xlim(min(time), max(time))
    ax1.set_ylim(-u0-0.05, u0+0.05)
    print(ds1.UVEL.isel(time=t,YC=40,XG=0,Z=0))
    point, = ax1.plot([ds1['time'].isel(time=t)/np.timedelta64(1, 's')/3600],[sur_bund.UVEL.isel(time=t,YC=40,XG=0,Z=0).data], 'go', color='red',ms=10)
    ax1.annotate('time [hr]',(85,-0.15))
    ax1.set_ylabel('U [m/s]')
    ax1.set_title('Forcing', loc='left')

    #fig2 U at diff Y
    fig.suptitle("t= %2.2f hrs" %time[t])
    for i in range(len(iy)):
        ax=fig.add_subplot(grid[1,i])
        print(ds1['XG0'].shape)
        print(ds1['Z'].shape)
        um=np.amax((ds1.UVEL.where(ds1['maskW'] !=0,np.nan)).isel(time=t,YC=iy[i]))
        ir = np.where((ds1.UVEL.where(ds1['maskW'] !=0,np.nan)).isel(time=t,YC=iy[i]) == um)
        icord = list(zip(ir[0], ir[1]))
        print(icord)
        xm=ds1['XG0'][ir[1]].values
        zm=ds1['Z'][ir[0]].values


        pcm=ax.pcolormesh(ds1['XG0'],ds1['Z'],(ds1.UVEL.where(ds1['maskW'] !=0,np.nan)).isel(time=t,YC=iy[i]),vmax=vmax,vmin=vmin,cmap='RdBu_r',rasterized=True)
        ax.contour(ds1['XC0'], ds1['Z'],ds1.THETA.isel(time=t,YC=iy[i]),np.sort(tRef[::4]),colors='0.3',linewidths=1)
        ax.plot(ds1['XC0'],-ds1.Depth.isel(YC=iy[i]),lw=1)
        ax.set_xlim([-3.,3.])
        ax.set_xticklabels([])
        ax.annotate('vmax=%1.2f'%um,xy=(xm,zm),xytext=(xm+0.1,zm+0.1))
        if i==0:
            ax.set_ylabel('Z [m]')
            ax.set_title('U [m/s], YC= %1.1f km' %yc0[iy[i]],loc='left')
        else:
            ax.set_yticklabels([])
            ax.set_title('YC= %1.1f km' %yc0[iy[i]])

    
    #fig3 V at diff Y
    for i in range(len(iy)):
        ax=fig.add_subplot(grid[2,i])
        pcm=ax.pcolormesh(ds1['XC0'],ds1['Z'],(ds1.VVEL.where(ds1['maskS'] !=0,np.nan)).isel(time=t,YG=iy[i]),vmax=vmax,vmin=vmin,cmap='RdBu_r',rasterized=True)
        print('iy' +str(iy))
        print('pcm'+str(pcm))
        ax.contour(ds1['XC0'], ds1['Z'],ds1.THETA.isel(time=t,YC=iy[i]),np.sort(tRef[::4]),colors='0.3',linewidths=1)
        ax.plot(ds1['XC0'],-ds1.Depth.isel(YC=iy[i]),lw=1)
        ax.set_xlim([-3.,3.])
        if i==0:
            ax.set_ylabel('Z [m]')
            ax.set_xlabel('X [km]')
            ax.set_title('V [m/s]',loc='left')
        else:
            ax.set_yticklabels([])

    
    #fig4 U at diff X
    for i in range(len(ix)):
        ax=fig.add_subplot(grid[3,i])
        pcm=ax.pcolormesh(ds1['YC0'],ds1['Z'],(ds1.UVEL.where(ds1['maskW'] !=0,np.nan)).isel(time=t,XG=ix[i]),vmax=vmax,vmin=vmin,cmap='RdBu_r',rasterized=True)
        print('ix' +str(ix))
        print('pcm'+str(pcm))
        ax.contour(ds1['YC0'], ds1['Z'],ds1.THETA.isel(time=t,XC=ix[i]),np.sort(tRef[::4]),colors='0.3',linewidths=1)
        ax.plot(ds1['YC0'],-ds1.Depth.isel(XC=ix[i]),lw=1)
        ax.set_xlim([-1.5,1.5])
        ax.set_xticklabels([])
        if i==0:
            ax.set_ylabel('Z [m]')
            ax.set_title('U [m/s], XC= %1.1f km' %xc0[ix[i]],loc='left')
        else:
            ax.set_yticklabels([])
            ax.set_title('XC= %1.1f km' %xc0[ix[i]])

    #fig5 V at diff X
    for i in range(len(ix)):
        ax=fig.add_subplot(grid[4,i])
        pcm=ax.pcolormesh(ds1['YG0'],ds1['Z'],(ds1.VVEL.where(ds1['maskS'] !=0,np.nan).isel(time=t,XC=ix[i])),vmax=vmax,vmin=vmin,cmap='RdBu_r',rasterized=True)
        print('ix' +str(ix))
        print('pcm'+str(pcm))
        ax.contour(ds1['YC0'], ds1['Z'],ds1.THETA.isel(time=t,XC=ix[i]),np.sort(tRef[::4]),colors='0.3',linewidths=1)
        ax.plot(ds1['YC0'],-ds1.Depth.isel(XC=ix[i]),lw=1)
        ax.set_xlim([-1.5,1.5])
        if i==0:
            ax.set_ylabel('Z [m]')
            ax.set_xlabel('Y [km]')
            ax.set_title('V [m/s]',loc='left')
        else:
            ax.set_yticklabels([])

    cax2 = fig.add_subplot(grid[1:5,-1])
    cax2.set_visible(False)
    fig.colorbar(pcm, ax=cax2,extend='both')

    #fig6 UV at diff Z

    for i in range(len(iz)):
        ax=fig.add_subplot(grid[5,i])
        pcm=ax.pcolormesh(ds1['XC0'],ds1['YC0'],ds1.THETA.isel(time=t,Z=iz[i]),vmax=40,vmin=tmin,cmap='RdBu_r',rasterized=True)
        UC=(Grid.interp(ds1.UVEL.isel(time=t,Z=iz[i]),'X',boundary='extrapolate')).where(ds1['maskC'].isel(Z=iz[i])!=0,np.nan)
        VC=(Grid.interp(ds1.VVEL.isel(time=t,Z=iz[i]),'Y',boundary='extrapolate')).where(ds1['maskC'].isel(Z=iz[i])!=0,np.nan)
        print(UC.shape)
        grid_U = griddata(points,UC.values.flatten(),(grid_x, grid_y))
        grid_V = griddata(points,VC.values.flatten(),(grid_x, grid_y))
        quiv=ax.quiver(grid_x0, grid_y0, grid_U, grid_V, units='xy', color='limegreen')
        ax.quiverkey(quiv,0.8,0.85,1,r'$1 \frac{m}{s}$', labelpos='E')
        ax.set_xlim([-3.,3.])
        ax.set_ylim([-1.5,1.5])
        ax.set_title('Z= %1.2f m' %z[iz[i]])
        if i==0:
            ax.set_xlabel('X [km]')
            ax.set_ylabel('Y [km]')
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
    
    cax3 = fig.add_subplot(grid[5,-1])
    cax3.set_visible(False)
    cbar=fig.colorbar(pcm, ax=cax3,extend='both')
    cbar.ax.set_ylabel('Temperature [$^o$C]')

    #plt.show()
    plt.savefig('./figsMag2/Background_A4_i%03d.png' %(t+120))
    fig.clf()
#plt.show()


