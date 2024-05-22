from xmitgcm import open_mdsdataset
data_dir1 = '/Users/jiaxuanchang/MITgcmExampleSteadyGauss-master_sx100/runs/RunFr1300/'
ds1 = open_mdsdataset(data_dir1, geometry='cartesian', endian='<')

data_dir2 = '/Users/jiaxuanchang/MITgcmExampleSteadyGauss-master/runs/RunFr1300/'
ds2 = open_mdsdataset(data_dir2, geometry='cartesian', endian='<')


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
import matplotlib.animation as animation

t = 0
xmin = 195
xmax = 220
vmax = 0.5
vmin = -0.5
wmax = 0.2
wmin = -0.2
tmax = 30
tmin = 0

#plot dx
fig = plt.figure(figsize=(20, 10))
grid = fig.subplots(nrows=2, ncols=3,  gridspec_kw={'width_ratios':(1,1,0.05), 'height_ratios':(0.85,0.9)})
*axes1, cax1 = grid[0]
*axes2, cax2 = grid[1]
cax1.set_visible(False)
#fig.subplots_adjust(hspace=0.3)
#fig.suptitle("t", fontsize=12)

axes1[0].plot(ds1.XC.values[:-1]/1e3, np.diff(ds1.XC.values))
#axes1[0].set_xlabel('X [km]')
axes1[0].set_ylabel('dx [m]')
dxmin1 = min(np.diff(ds1.XC.values))
axes1[0].text(205,500,'$dx_{min}=%2.2fm$' %dxmin1)
axes1[0].set(xlim=(xmin, xmax),ylim=(0,4000))
axes1[0].grid(True)
pos0 = axes1[0].get_position()

axes1[1].plot(ds2.XC.values[:-1]/1e3, np.diff(ds2.XC.values))
#axes1[1].set_xlabel('X [km]')
axes1[1].set_ylabel('dx [m]')
dxmin2 = min(np.diff(ds2.XC.values))
axes1[1].text(205,500,'$dx_{min}=%2.2fm$' %dxmin2)
axes1[1].set(xlim=(xmin, xmax),ylim=(0,4000))
axes1[1].grid(True)
pos1 = axes1[1].get_position()

#plot u velocity
cmap = cm.RdBu_r
levels = np.linspace(vmin, vmax, num=21)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
pcolor_opts1 = {'cmap':cm.get_cmap(cmap, len(levels) - 1), 'norm': norm}

T01 = np.squeeze(ds1.THETA.values[0, :, :, :])
TRef1 = ds1.THETA.values[0,:,0,0]
UVEL1 = np.squeeze(ds1.UVEL.values[t, :, :, :])
UVEL1 = np.ma.masked_where(T01 == 0, UVEL1)
X1, Z1 = np.meshgrid(ds1.UVEL.XG, ds1.UVEL.Z)

cset1 = axes2[0].pcolormesh(X1/1e3, Z1, UVEL1[:-1,:-1]-0.32, **pcolor_opts1)
#fig.colorbar(cset1, ax=axes2[0],ticks=np.arange(vmin, vmax+0.1, 0.5))
axes2[0].set_xlabel('X [km]')
axes2[0].set_ylabel('Z [m]')
axes2[0].set_title('U [m/s] - Ub', loc='left')
axes2[0].set(xlim=(xmin, xmax))

THETA1 = np.squeeze(ds1.THETA.values[t, :, :, :])
X1, Z1 = np.meshgrid(ds1.THETA.XC, ds1.THETA.Z)
axes2[0].contour(X1/1e3, Z1, THETA1, np.sort(TRef1[::2]),colors='0.3',)  #plot every two isotherms
line1 = axes2[0].plot(ds1.Depth.XC.values/1e3,-np.transpose(ds1.Depth.values))[0]
pos2 = axes2[0].get_position(original=False)


T02 = np.squeeze(ds2.THETA.values[0, :, :, :])
TRef2 = ds2.THETA.values[0,:,0,0]
UVEL2 = np.squeeze(ds2.UVEL.values[t, :, :, :])
UVEL2 = np.ma.masked_where(T02 == 0, UVEL2)
X2, Z2 = np.meshgrid(ds2.UVEL.XG, ds2.UVEL.Z)

cset2 = axes2[1].pcolormesh(X2/1e3, Z2, UVEL2[:-1,:-1]-0.32, **pcolor_opts1)

axes2[1].set_xlabel('X [km]')
axes2[1].set_ylabel('Z [m]')
axes2[1].set_title('U [m/s] - Ub', loc='left')
axes2[1].set(xlim=(xmin, xmax))

THETA2 = np.squeeze(ds2.THETA.values[t, :, :, :])
X2, Z2 = np.meshgrid(ds2.THETA.XC, ds2.THETA.Z)
axes2[1].contour(X2/1e3, Z2, THETA2, np.sort(TRef2[::2]),colors='0.3',)  #plot every two isotherms
line2 = axes2[1].plot(ds2.Depth.XC.values/1e3,-np.transpose(ds2.Depth.values))[0]
pos3 = axes2[1].get_position(original=False)

fig.colorbar(cset2, cax=cax2,ticks=np.arange(vmin, vmax+0.1, 0.5),shrink=0.8)
#axes1[0].set_position([pos2.x0, pos0.y0, pos2.width, pos0.height])
#axes1[1].set_position([pos3.x0, pos1.y0, pos3.width, pos1.height])
axes1[1].set_title('t= %2.3fhr' %0.)
plt.tight_layout()


#
def updateData(tt):
    X1, Z1 = np.meshgrid(ds1.THETA.XC, ds1.THETA.Z)
    U1 = np.squeeze(ds1.UVEL.values[tt,:,:,:])
    U1 = np.ma.masked_where(T01 == 0, U1)
    T1 = np.squeeze(ds1.THETA.values[tt,:,:,:])
    T1 = np.ma.masked_where(T01 == 0, T1)
    time1 = ds1.time.values[tt]/3600
    cset1.set_array(U1[:-1,:-1].flatten() - 0.32)
    axes2[0].collections[1:]=[]
    axes2[0].contour(X1/1e3, Z1, T1, np.sort(TRef1[::2]),colors='0.3')
    line1.set_ydata(-np.transpose(ds1.Depth.values))
    axes1[0].set_title('t= %2.3fhr' % time1)

    X2, Z2 = np.meshgrid(ds2.THETA.XC, ds2.THETA.Z)
    U2 = np.squeeze(ds2.UVEL.values[tt, :, :, :])
    U2 = np.ma.masked_where(T02 == 0, U2)
    T2 = np.squeeze(ds2.THETA.values[tt, :, :, :])
    T2 = np.ma.masked_where(T02 == 0, T2)
    time2 = ds2.time.values[tt] / 3600
    cset2.set_array(U2[:-1, :-1].flatten() - 0.32)
    axes2[1].collections[1:]=[]
    axes2[1].contour(X2/1e3, Z2, T2, np.sort(TRef2[::2]), colors='0.3')
    line2.set_ydata(-np.transpose(ds2.Depth.values))

    axes1[1].set_title('t= %2.3fhr' % time2)



simulation = animation.FuncAnimation(fig, updateData, blit=False, frames=200, interval=20, repeat=False)
plt.draw()
plt.show()
simulation.save(filename='comprdx_UT.gif', writer='imagemagick', fps=5)