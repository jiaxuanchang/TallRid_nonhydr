from xmitgcm import open_mdsdataset
import xgcm
import psyplot.project as psy
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import pandas as pd
import numpy as np
import math
import xarray as xr
import string

# Define function for string formatting of scientific notation
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(math.floor(math.log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds1 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energyvars','statevars','statevars2d'])
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'])
Grid = xgcm.Grid(ds1, periodic=False)

t = 0
h0 = 140.
H = 200.
f0 = 1.e-4
g = 9.8
u0=0.12

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


xmin = 30
xmax = 54
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


cmap = cm.RdBu_r
levels = np.linspace(vmin, vmax, num=numcolv)
Tlevels = np.linspace(tmin, tmax, num=numcolt)
#norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#pcolor_opts1 = {'cmap':cm.get_cmap(cmap, len(levels) - 1), 'norm': norm}

   
xg=ds1.coords["XG"]
xc=ds1.coords["XC"]
yg=ds1.coords["YG"]
yc=ds1.coords["YC"]
z=ds1.coords["Zl"]
time=ds1.coords["time"]/3600
print(ds1)
print(yc)
print(xc[185:216])
print(xc[-2:])
print(ds1.XC.mean())

ds1=ds1.assign_coords(XC0 = (ds1.XC/1000.-ds1.XC.mean()/1000.))
ds1=ds1.assign_coords(XG0 = (ds1.XG-ds1.XC.mean())/1000.)
ds1=ds1.assign_coords(YC0 = (ds1.YC/1000.-ds1.YC.mean()/1000.))
ds1=ds1.assign_coords(YG0 = (ds1.YG-ds1.YC.mean())/1000.)
yc0 = (ds1.YC/1000.-ds1.YC.mean()/1000.)
xc0 = (ds1.XC/1000.-ds1.XC.mean()/1000.)

fig = plt.figure(figsize=(13,9))
grid = plt.GridSpec(4, 6, hspace=0.8, height_ratios=[1.2,1,1, 1], width_ratios=[1,1,1,1,1,0.5])

# selece plotting slices
sx=[40610, 41410,42010,42610,43410] 
sy=[625.,1225.,1525.,1825.,2425.]
#sz=[-2.5,-17.5,-32.5,-47.5,-62.5]#Z
sz=[-5.,-20.,-35.,-50.,-65.]#Zl
iy=[]
iz=[]
ix=[]
for i in range(len(sy)):
    if sy[i] in yc:
        iy.append(list(yc.data).index(sy[i]))
        print(iy)

for i in range(len(sx)):
    if sx[i] in xc:
        ix.append(list(xc.data).index(sx[i]))
        print(ix)

for i in range(len(sz)):
    if sz[i] in z:
        iz.append(list(z.data).index(sz[i]))
        print(iz)

dmax=10**(-2)
dmin=10**(-10)

for t in range(100,180):#range(ttlen):
    fig.clf()
    ax1=fig.add_subplot(grid[0,:-1])
    eps=ds1['KLeps'].isel(time=t)#Zl,YC,XC
    T=ds1['THETA'].isel(time=t).where(ds1['maskC']!=0,np.nan)
    Dep=-ds1['Depth']

    #fig1 forcing
    ax1.plot(time,np.zeros(ttlen),color='gray',linestyle='--')
    ax1.plot(time,ds1['UVEL'].isel(YC=2, XG=0, Z=0),label='U(t,0,0,0)')
    ax1.set_xlim(min(time), max(time))
    ax1.set_ylim(-u0-0.05, u0+0.05)
    print(ds1['UVEL'].isel(time=t,YC=2, XG=0, Z=0).values)
    point, = ax1.plot([ds1['time'].isel(time=t)/3600],[ds1['UVEL'].isel(time=t,YC=2, XG=0, Z=0)], 'go', color='red',ms=10)
    ax1.annotate('time [hr]',(85,-0.15))
    ax1.set_ylabel('U [m/s]')
    ax1.set_title('Forcing', loc='left')

    #fig2 U at diff Y
    fig.suptitle("t= %2.2f hrs" %time[t])
    for i in range(len(iy)):
        ax=fig.add_subplot(grid[1,i])
        eplt=eps.isel(YC=iy[i])
        dplt=Dep.isel(YC=iy[i])
        print(ds1['XG0'].shape)
        print(ds1['Z'].shape)
        print(eplt.shape)
        em=np.amax(eplt.values)
        ir = np.where(eplt == em)
        icord = list(zip(ir[0], ir[1]))
        print(icord)
        xm=ds1['XC0'][ir[1]].values
        zm=ds1['Zl'][ir[0]].values


        pcm=ax.pcolormesh(ds1['XC0'],ds1['Zl'],eplt,vmax=dmax,vmin=dmin
                ,norm=colors.LogNorm(vmin=dmin, vmax=dmax),cmap='RdYlBu_r',rasterized=True)
        ax.contour(ds1['XC0'], ds1['Z'],T.isel(YC=iy[i]),np.sort(tRef[::4]),colors='0.3',linewidths=1)
        ax.plot(ds1['XC0'],dplt,lw=1)
        ax.set_xlim([-2.,2.])
        ax.set_xticklabels([])
        ax.annotate('max= %.1E' %em,xy=(xm,zm),xytext=(xm-0.15,zm+0.1))
        if i==0:
            ax.set_ylabel('Z [m]')
        else:
            ax.set_yticklabels([])
            ax.set_title('YC= %1.1f km' %yc0[iy[i]])

    
    #fig3 U at diff X
    for i in range(len(ix)):
        ax=fig.add_subplot(grid[2,i])
        pcm=ax.pcolormesh(ds1['YC0'],ds1['Zl'],eps.isel(XC=ix[i]),vmax=dmax,vmin=dmin
                ,norm=colors.LogNorm(vmin=dmin, vmax=dmax),cmap='RdYlBu_r',rasterized=True)
        print('ix' +str(ix))
        print('pcm'+str(pcm))
        ax.contour(ds1['YC0'], ds1['Z'],T.isel(XC=ix[i]),np.sort(tRef[::4]),colors='0.3',linewidths=1)
        ax.plot(ds1['YC0'],Dep.isel(XC=ix[i]),lw=1)
        ax.set_xlim([-1.5,1.5])
        ax.set_xticklabels([])
        if i==0:
            ax.set_ylabel('Z [m]')
        else:
            ax.set_yticklabels([])
            ax.set_title('XC= %1.1f km' %xc0[ix[i]])

    #fig4 UV at diff Z
    for i in range(len(iz)):
        ax=fig.add_subplot(grid[3,i])
        pcm=ax.pcolormesh(ds1['XC0'],ds1['YC0'],eps.isel(Zl=iz[i]),vmax=dmax,vmin=dmin
                ,norm=colors.LogNorm(vmin=dmin, vmax=dmax),cmap='RdYlBu_r',rasterized=True)
        ax.set_xlim([-2.,2.])
        ax.set_ylim([-1.5,1.5])
        ax.set_title('Z= %1.2f m' %z[iz[i]])
        if i==0:
            ax.set_xlabel('X [km]')
            ax.set_ylabel('Y [km]')
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
    
    cax = fig.add_subplot(grid[1:4,-1])
    cax.set_visible(False)
    cbar=fig.colorbar(pcm, ax=cax,extend='both')
    cbar.ax.set_ylabel('Dissipation rate')

    #plt.show()
    plt.savefig('./figsMag2/Bk_eps_A4_i%03d.png' %t)

#plt.show()


