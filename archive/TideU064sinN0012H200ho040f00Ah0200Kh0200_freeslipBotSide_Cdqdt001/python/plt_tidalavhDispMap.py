from xmitgcm import open_mdsdataset
import xgcm
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import math
import string
import dask.array as da

import warnings

warnings.filterwarnings('ignore')

def nansumwrapper(a, **kwargs):
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nansum(a, **kwargs)


currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

ds1 = open_mdsdataset(data_dir, geometry='cartesian',endian='<',prefix=['statevars','statevars2d'])

grid = xgcm.Grid(ds1, periodic=False)
print(grid)

t = 0
f0 = 1.e-4
g = 9.8
rhoNil=999.8
viscAh=1.e-4

om=2*np.pi/12.4/3600
alpha = 2e-4
beta = 0.
nz = 200
#dz=H/nz 
tR_fname="../indata/TRef.bin"
tRef = np.fromfile(tR_fname)
refSalt=35.

xmin = -27
xmax = 27
ymin = 200
ymax = 2800
numcolt=21
numcolv=21
                                        
ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )
print('initial temp: '+ str(tRef))

time=ds1.coords['time'].values/np.timedelta64(1, 's')
xc=ds1.coords['XC']
xg=ds1.coords['XG']
yc=ds1.coords['YC']
yg=ds1.coords['YG']
z=ds1.coords['Z']
zl=ds1.coords['Zl']

xc0 = (xc/1000.-xc.mean()/1000.)
xg0 = (xg/1000.-xc.mean()/1000.)
yc0 = (yc/1000.-yc.mean()/1000.)
yg0 = (yg/1000.-yc.mean()/1000.)



iix=190
print(xc[iix])
print('Depth[:,iix]:' + str(ds1.Depth[:,iix].values))
print((ds1.drF*ds1.hFacC*ds1.maskC).sum('Z')[:,iix].values)

if 1:
    dt=time[1]-time[0]
    print(dt)
    
    T=7

    time_bin_labels = np.arange(12.4*60*60/(T+0.5),time[-1]-20000,12.4*60*60)
    print(time_bin_labels)
    t_st=int(12.4*60*60*T/dt+1)
    t_en=int(12.4*60*60*(T+1)/dt+1)
    timestd=np.arange(12.4*60*60*T+dt,12.4*60*60*(T+1)+dt,dt)

    hDbc_dudx_t=np.zeros((t_en-t_st,len(yc),len(xc)),dtype=float)
    hDbc_dvdy_t=np.zeros((t_en-t_st,len(yc),len(xc)),dtype=float)
    hDbc_dvdx_t=np.zeros((t_en-t_st,len(yg),len(xg)),dtype=float)
    hDbc_dudy_t=np.zeros((t_en-t_st,len(yg),len(xg)),dtype=float)
    hDbc_dwdx_t=np.zeros((t_en-t_st,len(yc),len(xg)),dtype=float)
    hDbc_dwdy_t=np.zeros((t_en-t_st,len(yg),len(xc)),dtype=float)
    hDbc_t=np.zeros((t_en-t_st,len(yc),len(xc)),dtype=float)
    iiy=35

    for tt in range(t_st,t_en):
        print(tt)
        US=grid.interp(ds1.UVEL.isel(time=tt),axis=('X','Y'),boundary='extrapolate')
        VW=grid.interp(ds1.VVEL.isel(time=tt),axis=('X','Y'),boundary='extrapolate')
        print('U[tt,:,iiy,iix]:' +str(ds1.UVEL.isel(time=tt,YC=iiy,XG=iix).values))
        print('V[tt,:,iiy,iix]:' +str(ds1.VVEL.isel(time=tt,YG=iiy,XC=iix).values))
        print('depth[iiy,iix]:' +str(ds1.Depth.isel(YC=iiy,XC=iix).values))
        print('DepW[iiy,iix]' +str((ds1.drF*ds1.hFacW*ds1.maskW).sum('Z').isel(YC=iiy,XG=iix).values))
        print('DepS[iiy,iix]' +str((ds1.drF*ds1.hFacS*ds1.maskS).sum('Z').isel(YG=iiy,XC=iix).values))
        print((ds1['drF']*ds1['hFacW']*ds1['maskW']).isel(YC=iiy,XG=iix).values)

        #depth mean velocity
        U0W=(ds1['UVEL'].isel(time=tt)*ds1['drF']*ds1['hFacW']*ds1['maskW']).sum('Z') \
                /(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
        V0S=(ds1['VVEL'].isel(time=tt)*ds1['drF']*ds1['hFacS']*ds1['maskS']).sum('Z') \
                /(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
        print('U0W[iiy,iix]'+str(U0W[iiy,iix].values))
        print('V0S[iiy,iix]'+str(V0S[iiy,iix].values))

        U0S=(US*ds1['drF']*ds1['hFacS']*ds1['maskS']).sum('Z')\
                /(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
        V0W=(VW*ds1['drF']*ds1['hFacW']*ds1['maskW']).sum('Z')\
                /(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
        print('US[iiy,iix]'+str(U0S[iiy,iix].values))
        print('VW[iiy,iix]'+str(V0W[iiy,iix].values))
        

        #baroclinic velocity
        ds1['maskL']=grid.interp(ds1.hFacC,'Z',to='left', boundary='extrapolate')
        upW=(ds1['UVEL'].isel(time=tt)-U0W).where(ds1['maskW'] !=0,np.nan)   #mask
        vpS=(ds1['VVEL'].isel(time=tt)-V0S).where(ds1['maskS'] !=0,np.nan)   #mask 
        upS=(US-U0S).where(ds1['maskS'] !=0,np.nan)  #mask
        vpW=(VW-V0W).where(ds1['maskW'] !=0,np.nan)  #mask
        wp=ds1['WVEL'].isel(time=tt).where(ds1['maskL'] !=0, np.nan)
        windowsize=2


        print('upW[:,iiy,iix]:'+str(upW.isel(YC=iiy,XG=iix).values))
        print('vpS[:,iiy,iix]:'+str(vpS.isel(YG=iiy,XC=iix).values))
        

        print('maskW(z=0)' +str(ds1['maskW'][0,:,iix].values))
        print('sum of upw:' + str((upW*ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')[:,iix].values))
        print('sum of vpw:' + str((vpW*ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')[:,iix].values))

        print('maskS(z=0)' +str(ds1['maskS'][0,:,iix].values))
        print('sum of ups:' + str((upS*ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')[:,iix].values))
        print('sum of vps:' + str((vpS*ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')[:,iix].values))
        

        # Depth-integrated horizontal dissipation
        ds1['maskZ'] = grid.interp(ds1.hFacS, 'X', boundary='extrapolate')
        ds1['hFacZ'] = grid.interp(ds1.hFacS, 'X', boundary='extrapolate')
        ds1['hFacWL'] = grid.interp(ds1.hFacW, 'Z', to='left', boundary='extrapolate')
        print(ds1.hFacWL.dims)
        ds1['hFacSL'] = grid.interp(ds1.hFacS, 'Z', to='left', boundary='extrapolate')
        print(ds1.hFacSL.dims)
        ds1['drL']=grid.interp(ds1.drF, 'Z', to='left', boundary='extrapolate')
        print(ds1.drL.dims)
        
        hDbc_dudx = rhoNil*viscAh*((((grid.diff(upW* ds1.dyG , 'X'
            , boundary='extrapolate')/ds1.rA)**2)*(ds1['drF']*ds1['hFacC'])) \
                    .where(ds1['maskC'] !=0,np.nan)).sum('Z',skipna=True,min_count=1)
        hDbc_dvdy = rhoNil*viscAh*((((grid.diff(vpS* ds1.dxG , 'Y'
            , boundary='extrapolate')/ds1.rA)**2)*(ds1['drF']*ds1['hFacC'])) \
                    .where(ds1['maskC'] !=0,np.nan)).sum('Z',skipna=True,min_count=1)
        hDbc_dvdx = rhoNil*viscAh*((((grid.diff(vpS* ds1.dyC , 'X' 
            , boundary='extrapolate')/ds1.rAz)**2)*(ds1['drF']*ds1['hFacZ'])) \
                    .where(ds1['maskZ'] !=0,np.nan)).sum('Z',skipna=True,min_count=1)
        hDbc_dudy = rhoNil*viscAh*((((grid.diff(upW* ds1.dxC , 'Y'
            , boundary='extrapolate')/ds1.rAz)**2)*(ds1['drF']*ds1['hFacZ'])) \
                    .where(ds1['maskZ'] !=0,np.nan)).sum('Z',skipna=True,min_count=1)
        hDbc_dwdx = rhoNil*viscAh*((((grid.diff(wp ,'X',boundary='extrapolate')/ds1.dxC) \
                **2)*(ds1['drL']*ds1['hFacWL'])).where(ds1['hFacWL'] !=0,np.nan)).sum('Zl',skipna=True,min_count=1)
        hDbc_dwdy = rhoNil*viscAh*((((grid.diff(wp ,'Y',boundary='extrapolate')/ds1.dyC) \
                **2)*(ds1['drL']*ds1['hFacSL'])).where(ds1['hFacSL'] !=0,np.nan)).sum('Zl',skipna=True,min_count=1)


        print('dudx' +str(hDbc_dudx[:,iix].values))
        
        norm=colors.LogNorm(vmin=1e-6, vmax=1e-3)       
        
        if 1:
            
            Dep=ds1['Depth']
            levels=range(0,210,50)
            f4, axes = plt.subplots(figsize=(15,9.5), nrows=3, ncols=2, sharex='all', sharey='all')
            axes=axes.flatten()
            f4.subplots_adjust(top=0.9,left=0.08,right=0.95)
            f4.suptitle("t= %d s" %time[tt], size=16)
            
            p1=axes[0].pcolormesh(xc0,yc0,hDbc_dudx,norm=norm)
            f4.colorbar(p1,ax=axes[0])
            axes[0].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
            axes[0].set_title(r'$ \overline{ \rho_{0} \nu_{H} (\frac{\partial u^{\prime}}{\partial x})^2  }$')
            axes[0].set_xlim(xmin,xmax)
            axes[0].set_xlabel('')
            axes[0].set_ylabel('y coordinate [km]')

            p2=axes[1].pcolormesh(xc0,yc0,hDbc_dvdy,norm=norm)
            f4.colorbar(p2,ax=axes[1])
            axes[1].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
            axes[1].set_title(r'$\ \overline{ \rho_{0} \nu_{H} (\frac{\partial v^{\prime}}{\partial y})^2  }$')
            axes[1].set_xlim(xmin,xmax)
            axes[1].set_xlabel('')
            
            p3=axes[2].pcolormesh(xg0,yg0,hDbc_dvdx,norm=norm)
            f4.colorbar(p3,ax=axes[2])
            axes[2].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
            axes[2].set_title(r'$\ \overline{ \rho_{0} \nu_{H} (\frac{\partial v^{\prime}}{\partial x})^2  }$')
            axes[2].set_xlim(xmin,xmax)
            axes[2].set_xlabel('')
            axes[2].set_ylabel('y coordinate [km]')

            p4=axes[3].pcolormesh(xg0,yg0,hDbc_dudy,norm=norm)
            f4.colorbar(p4,ax=axes[3])
            axes[3].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
            axes[3].set_title(r'$\ \overline{ \rho_{0} \nu_{H} (\frac{\partial u^{\prime}}{\partial y})^2  }$')
            axes[3].set_xlim(xmin,xmax)
            axes[3].set_xlabel('')

            p5=axes[4].pcolormesh(xg0,yc0,hDbc_dwdx,norm=norm)
            f4.colorbar(p5,ax=axes[4])
            axes[4].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
            axes[4].set_title(r'$\ \overline{ \rho_{0} \nu_{H} (\frac{\partial w^{\prime}}{\partial x})^2  }$')
            axes[4].set_xlim(xmin,xmax)
            axes[4].set_ylabel('y coordinate [km]')
            axes[4].set_xlabel('x coordinate [km]')

            p6=axes[5].pcolormesh(xc0,yg0,hDbc_dwdy,norm=norm)
            f4.colorbar(p6,ax=axes[5])
            axes[5].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
            axes[5].set_title(r'$\ \overline{ \rho_{0} \nu_{H} (\frac{\partial w^{\prime}}{\partial y})^2  }$')
            axes[5].set_xlim(xmin,xmax)
            axes[5].set_xlabel('x coordinate [km]')
            
            
            plt.savefig('./figsMag/EnergyBudgethDsp_x27y_t%d_Log.png' %tt)
            plt.close(f4)


        hDbc_dudx_t[tt-t_st,:,:]=hDbc_dudx
        hDbc_dvdy_t[tt-t_st,:,:]=hDbc_dvdy
        hDbc_dvdx_t[tt-t_st,:,:]=hDbc_dvdx
        hDbc_dudy_t[tt-t_st,:,:]=hDbc_dudy
        hDbc_dwdx_t[tt-t_st,:,:]=hDbc_dwdx    
        hDbc_dwdy_t[tt-t_st,:,:]=hDbc_dwdy    
        

    #print(hDbc_t[:,10:15,90:93])
    print(hDbc_dwdy_t.shape)
    
    # tidal average
    ta_hDbc_dudx=hDbc_dudx_t.mean(axis=0)
    ta_hDbc_dvdy=hDbc_dvdy_t.mean(axis=0)
    print(ta_hDbc_dvdy.shape)
    ta_hDbc_dvdx=hDbc_dvdx_t.mean(axis=0)
    ta_hDbc_dudy=hDbc_dudy_t.mean(axis=0)
    ta_hDbc_dwdx=hDbc_dwdx_t.mean(axis=0)
    ta_hDbc_dwdy=hDbc_dwdy_t.mean(axis=0)
    print(ta_hDbc_dwdx[:,iix])


    ixc=np.where(np.logical_and(xc0>xmin, xc0<=xmax))[0]
    iyc=np.where((yc > ymin) & (yc < ymax))[0]
    ixg=np.where(np.logical_and(xg0>xmin, xg0<=xmax))[0]
    iyg=np.where((yg > ymin) & (yg < ymax))[0]
    print(ixc[0])
    print(ixc[-1])
    print(ixg[0])
    print(ixg[-1])
    print(iyc[0])
    print(iyc[-1])
    #print(ta_hDbc[iyc[0]][ixc[0]])
    #print(ta_hDbc[iyc[0]][ixc[-1]])
    #print(ta_hDbc[iyc[-1]][ix[0]])
    #print(ta_hDbc[iy[-1]][ix[-1]])
    
    # Area integration
    hDspbc_dudx=nansumwrapper(ta_hDbc_dudx[iyc[0]:iyc[-1]+1,ixc[0]:ixc[-1]+1]*(ds1.rA[iyc[0]:iyc[-1]+1,ixc[0]:ixc[-1]+1].values))/1e6
    hDspbc_dvdy=nansumwrapper(ta_hDbc_dvdy[iyc[0]:iyc[-1]+1,ixc[0]:ixc[-1]+1]*(ds1.rA[iyc[0]:iyc[-1]+1,ixc[0]:ixc[-1]+1].values))/1e6
    hDspbc_dvdx=nansumwrapper(ta_hDbc_dvdx[iyg[0]:iyg[-1]+1,ixg[0]:ixg[-1]+1]*(ds1.rAz[iyg[0]:iyg[-1]+1,ixg[0]:ixg[-1]+1].values))/1e6
    hDspbc_dudy=nansumwrapper(ta_hDbc_dudy[iyg[0]:iyg[-1]+1,ixg[0]:ixg[-1]+1]*(ds1.rAz[iyg[0]:iyg[-1]+1,ixg[0]:ixg[-1]+1].values))/1e6
    hDspbc_dwdx=nansumwrapper(ta_hDbc_dwdx[iyc[0]:iyc[-1]+1,ixg[0]:ixg[-1]+1]*(ds1.rAw[iyc[0]:iyc[-1]+1,ixg[0]:ixg[-1]+1].values))/1e6
    hDspbc_dwdy=nansumwrapper(ta_hDbc_dwdy[iyg[0]:iyg[-1]+1,ixc[0]:ixc[-1]+1]*(ds1.rAs[iyg[0]:iyg[-1]+1,ixc[0]:ixc[-1]+1].values))/1e6
    

    print('hDspbc: (dudx)^2 - %f' %hDspbc_dudx )  
    print('hDspbc: (dvdy)^2 - %f' %hDspbc_dvdy )  
    print('hDspbc: (dvdx)^2 - %f' %hDspbc_dvdx )  
    print('hDspbc: (dudy)^2 - %f' %hDspbc_dudy )  
    print('hDspbc: (dwdx)^2 - %f' %hDspbc_dwdx )  
    print('hDspbc: (dwdy)^2 - %f' %hDspbc_dwdy )  
    #print('hDspbc: (dudx)^2+(dvdy)^2+(dvdx)^2+(dudy)^2 - %f' %hDspbc )  
    
    Dep=ds1['Depth']
    levels=range(0,210,50)
    
    norm=colors.LogNorm(vmin=1e-6, vmax=1e-3)
    # plotting
    fig, axes = plt.subplots(figsize=(15,9.5), nrows=3, ncols=2, sharex='all', sharey='all')
    axes=axes.flatten()
    fig.subplots_adjust(top=0.9,left=0.08,right=0.95)
    fig.suptitle("t= %1d tidal" %T, size=16)
    
    p1=axes[0].pcolormesh(xc0,yc0,ta_hDbc_dudx,norm=norm)
    print(np.nanmin(ta_hDbc_dudx))
    print(np.nanmax(ta_hDbc_dudx))
    fig.colorbar(p1,ax=axes[0])
    axes[0].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
    axes[0].set_title(r'$\langle \overline{ \rho_{0} \nu_{H} (\frac{\partial u^{\prime}}{\partial x})^2  }\rangle$')
    axes[0].set_xlim(xmin,xmax)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('y coordinate [km]')
    
    p2=axes[1].pcolormesh(xc0,yc0,ta_hDbc_dudy,norm=norm)
    print(np.nanmin(ta_hDbc_dudy))
    print(np.nanmax(ta_hDbc_dudy))
    fig.colorbar(p2,ax=axes[1])
    axes[1].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
    axes[1].set_title(r'$\langle \overline{ \rho_{0} \nu_{H} (\frac{\partial u^{\prime}}{\partial y})^2  }\rangle$')
    axes[1].set_xlim(xmin,xmax)
    axes[1].set_xlabel('')
    
    p3=axes[2].pcolormesh(xg0,yg0,ta_hDbc_dvdx,norm=norm)
    print(np.nanmin(ta_hDbc_dvdx))
    print(np.nanmax(ta_hDbc_dvdx))
    fig.colorbar(p3,ax=axes[2])
    axes[2].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
    axes[2].set_title(r'$\langle \overline{ \rho_{0} \nu_{H} (\frac{\partial v^{\prime}}{\partial x})^2  }\rangle$')
    axes[2].set_xlim(xmin,xmax)
    axes[2].set_xlabel('')
    axes[2].set_ylabel('y coordinate [km]')

    p4=axes[3].pcolormesh(xg0,yg0,ta_hDbc_dvdy,norm=norm)
    print(np.nanmin(ta_hDbc_dvdy))
    print(np.nanmax(ta_hDbc_dvdy))
    fig.colorbar(p4,ax=axes[3])
    axes[3].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
    axes[3].set_title(r'$\langle \overline{ \rho_{0} \nu_{H} (\frac{\partial v^{\prime}}{\partial y})^2  }\rangle$')
    axes[3].set_xlim(xmin,xmax)
    axes[3].set_xlabel('')


    p5=axes[4].pcolormesh(xg0,yc0,ta_hDbc_dwdx,norm=norm)
    print(np.nanmin(ta_hDbc_dwdx))
    print(np.nanmax(ta_hDbc_dwdx))
    fig.colorbar(p5,ax=axes[4])
    axes[4].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
    axes[4].set_title(r'$\langle \overline{ \rho_{0} \nu_{H} (\frac{\partial w^{\prime}}{\partial x})^2  }\rangle$')
    axes[4].set_xlim(xmin,xmax)
    axes[4].set_xlabel('x coordinate [km]')
    axes[4].set_ylabel('y coordinate [km]')

    p6=axes[5].pcolormesh(xc0,yg0,ta_hDbc_dwdy,norm=norm)
    print(np.nanmin(ta_hDbc_dwdy))
    print(np.nanmax(ta_hDbc_dwdy))
    fig.colorbar(p6,ax=axes[5])
    axes[5].contour(xc0,yc0,Dep,levels=levels,colors='grey',linewidths=0.7)
    axes[5].set_title(r'$\langle \overline{ \rho_{0} \nu_{H} (\frac{\partial w^{\prime}}{\partial y})^2  }\rangle$')
    axes[5].set_xlim(xmin,xmax)
    axes[4].set_xlabel('x coordinate [km]')

    #plt.show()
    plt.savefig('./figsMag/EnergyBudgethDsp_x27y_tidalP7_Log.png')
    #plt.show()


