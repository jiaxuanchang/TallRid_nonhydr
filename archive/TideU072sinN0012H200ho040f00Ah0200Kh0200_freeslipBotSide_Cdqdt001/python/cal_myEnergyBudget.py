from xmitgcm import open_mdsdataset
import xgcm
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import pandas as pd
import numpy as np
import math
from scipy import integrate
import xarray as xr
import string

currentDirectory = os.getcwd()
data_dir = currentDirectory[:-7] + '/input/'
print(data_dir)

iT=range(6,8)
P=12.4*60*60
dt=1860
t_st=int(P*iT[0])
t_en=int(P*(iT[-1]))
iters=range(t_st,t_en,dt)
print(iters)


ds1 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energyvars','statevars','statevars2d'],iters=iters)
ds2 = open_mdsdataset(data_dir, geometry='cartesian', endian='<',prefix=['energymvars'],iters=iters)
print('ds1: ')
print(ds1)


grid = xgcm.Grid(ds1, periodic=False)
print(grid)

f0 = 1.e-4
g = 9.8
rhoNil=999.8
rhoConst=rhoNil

om=2*np.pi/12.4/3600
alpha = 2e-4
beta = 0e-4
nz = 200
#dz=H/nz 

tR_fname="../indata/TRef.bin"
tRef = np.fromfile(tR_fname)
refSalt=35.
refTemp=tRef[0]
print('refTemp='+ str(refTemp))
print('tRef='+ str(tRef))

rho2=rhoNil*(1-(alpha*(tRef-refTemp)))
rhoS=np.roll(rho2,1)
N2=g/rhoNil*(rho2-rhoS)/ds1['drF'].values
#print('N2='+str(N2))

N2[0]=g/rhoNil*(rhoS[0]-rho2[0])/ds1['drF'][0]
#print(N2)

ttlen=len(ds1.time)
print('the length of time:' + str(ttlen) )

time1=ds1.coords['time']
time2=ds2.coords['time']
xc=ds1.coords['XC']
xg=ds1.coords['XG']
yc=ds1.coords['YC']
yg=ds1.coords['YG']
z=ds1.coords['Z']

xmin = xc.mean()-60000
xmax = xc.mean()+60000
numcolt=21
numcolv=21

PS = grid.interp(ds1['PHIHYD'],axis='Y',boundary='extrapolate') 
PW = grid.interp(ds1['PHIHYD'],axis='X',boundary='extrapolate')

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

# this US.VW.UC.VC only correct in 2d cases
UC=grid.interp(ds1['UVEL'],axis='X',boundary='extrapolate')
US=grid.interp(UC,axis='Y',boundary='extrapolate')
VC=grid.interp(ds1['VVEL'],axis='Y',boundary='extrapolate')
VW=grid.interp(VC,axis='X',boundary='extrapolate')


#print('U:' + str(ds1['UVEL'].isel(YC=0,time=100,Z=2).values))
#print('US:' +str(US[100,2,0,:]))
#print('V:' +str(ds1['VVEL'].isel(YG=0,time=100,Z=2).values))
#print('VW:' + str(VW[100,2,0,:]))

#depth mean pressure
P0S=(PS*ds1['drF']*ds1['hFacS']*ds1['maskS']).sum('Z')/(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
P0W=(PW*ds1['drF']*ds1['hFacW']*ds1['maskW']).sum('Z')/(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
#print('P0W=' +str(P0W[100,0,:]))
print(np.shape(P0W))

#depth mean velocity
U0W=(ds1['UVEL']*ds1['drF']*ds1['hFacW']*ds1['maskW']).sum('Z')/(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
U0S=(US*ds1['drF']*ds1['hFacS']*ds1['maskS']).sum('Z')/(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
U0C=(UC*ds1['drF']*ds1['hFacC']*ds1['maskC']).sum('Z')/(ds1.drF*ds1.hFacC*ds1.maskC).sum('Z')
V0W=(VW*ds1['drF']*ds1['hFacW']*ds1['maskW']).sum('Z')/(ds1.drF*ds1.hFacW*ds1.maskW).sum('Z')
V0S=(ds1.VVEL*ds1['drF']*ds1['hFacS']*ds1['maskS']).sum('Z')/(ds1.drF*ds1.hFacS*ds1.maskS).sum('Z')
V0C=(VC*ds1['drF']*ds1['hFacC']*ds1['maskC']).sum('Z')/(ds1.drF*ds1.hFacC*ds1.maskC).sum('Z')
print('shape of UOW & U0S:')
print(np.shape(U0W))
print(np.shape(U0S))
#print(U0W[100,0,:])

#SIZE OF DEPTH MEAN VARIABLE:(TIME,Y,X)

ds1['N2']=xr.DataArray(N2,coords=[z],dims=['Z'])
ds1['tRef']=xr.DataArray(tRef,coords=[z],dims=['Z'])

ds1['upW']=(ds1['UVEL']-U0W)*ds1['maskW']
ds1['upS']=(US-U0S)*ds1['maskS']
ds1['upC']=(UC-U0C)*ds1['maskC']
ds1['vpW']=(VW-V0W)*ds1['maskW']
ds1['vpS']=(ds1['VVEL']-V0S)*ds1['maskS']
ds1['vpC']=(VC-V0C)*ds1['maskC']

iy=20
if 0:
    plt.clf()
    f, ax = plt.subplots(3, 1, figsize=(11,9) , sharey=True)
    ds1['upW'].isel(YC=iy).mean('time').plot(ax=ax[0],cmap='RdBu_r',vmax=1,vmin=-1,cbar_kwargs={"label": "", "aspect": 40})
    ax[0].set_title('upW')
    ds1['upS'].isel(YG=iy).mean('time').plot(ax=ax[1],cmap='RdBu_r',vmax=1,vmin=-1,cbar_kwargs={"label": "", "aspect": 40})
    ax[1].set_title('upS')
    ds1['upC'].isel(YC=iy).mean('time').plot(ax=ax[2],cmap='RdBu_r',vmax=1,vmin=-1,cbar_kwargs={"label": "", "aspect": 40})
    ax[2].set_title('upC')
    for n, axs in enumerate(ax):
        axs.text(-0.1, 1, string.ascii_lowercase[n], transform=axs.transAxes,
            size=20, weight='bold')
    plt.tight_layout()
    plt.savefig('./figs/myta_up_y%d.png' %iy)

if 0:
    plt.clf()
    f, ax = plt.subplots(3, 1, figsize=(11,9) , sharey=True)
    ds1['vpW'].isel(YC=iy).mean('time').plot(ax=ax[0],cmap='RdBu_r',vmax=1,vmin=-1,cbar_kwargs={"label": "", "aspect": 40})
    ax[0].set_title('vpW')
    ds1['vpS'].isel(YG=iy).mean('time').plot(ax=ax[1],cmap='RdBu_r',vmax=1,vmin=-1,cbar_kwargs={"label": "", "aspect": 40})
    ax[1].set_title('vpS')
    ds1['vpC'].isel(YC=iy).mean('time').plot(ax=ax[2],cmap='RdBu_r',vmax=1,vmin=-1,cbar_kwargs={"label": "", "aspect": 40})
    ax[2].set_title('vpC')
    for n, axs in enumerate(ax):
        axs.text(-0.1, 1, string.ascii_lowercase[n], transform=axs.transAxes,
            size=20, weight='bold')
    plt.tight_layout()
    plt.savefig('./figs/myta_vp_y%d.png' %iy)


#follow Jody's Diagnostic, let's pretend wVEL wC=wW=wS
wC=grid.interp(ds1.WVEL,'Z',boundary='extrapolate')
print('dims of wC :'+ str(wC.dims))

wW=xr.DataArray(wC.values,coords=[time1,z,yc,xg],dims=['time','Z','YC','XG'])
wS=xr.DataArray(wC.values,coords=[time1,z,yg,xc],dims=['time','Z','YG','XC'])

#presure work
ds1['uPbc']=(PW-P0W)*ds1['upW']*ds1['drF']*ds1['hFacW']*ds1['maskW']
ds1['vPbc']=(PS-P0S)*ds1['vpS']*ds1['drF']*ds1['hFacS']*ds1['maskS']
#print(ds1['uPbc'])
#print(ds1['vPbc'])

#kinetic energy
ds1['kEpW'] = 0.5*(ds1['upW']*ds1['upW']+ds1['vpW']*ds1['vpW'] + wW*wW*ds1['maskW']) #(6.9)
ds1['kEpS'] = 0.5*(ds1['upS']*ds1['upS']+ds1['vpS']*ds1['vpS'] + wS*wS*ds1['maskS'])
ds1['kEpC'] = 0.5*(ds1['upC']*ds1['upC']+ds1['vpC']*ds1['vpC'] + wC*wC*ds1['maskC'])
print('kEpW dims: '+str(ds1.kEpW.dims))


#potential energy
dRho=rhoNil-rhoConst
ds1['rhoC']=(rhoNil*(-alpha*(ds1['THETA']-ds1['tRef'])+beta*(ds1['SALT']-refSalt)))*ds1['maskC']+dRho
#print(ds1['THETA'][100,:,0,198].values)
#print(ds1['rhoC'][100,:,0,198].values)
ds1['Ep']=g*g*ds1['rhoC']*ds1['rhoC']/2/rhoNil/rhoNil/ds1['N2']
myEbc=ds1['kEpC']+ds1['Ep']
#print(kEpC[100,:,0,198].values)
#print(Ep[100,:,0,198].values)
#print(myEbc[100,:,0,198].values)

if 0:
    plt.clf()
    f, ax = plt.subplots(3, 1, figsize=(11,9) , sharey=True)
    #plt.figure(figsize=(15,6))
    
    ds1['kEpW'].sum('Z').mean('time').plot(ax=ax[0],cbar_kwargs={"label": "", "aspect": 40})
    ax[0].set_title('kEpW')
    ds1['kEpC'].sum('Z').mean('time').plot(ax=ax[1],cbar_kwargs={"label": "", "aspect": 40})
    ax[1].set_title('kEpC')
    ds1['kEpS'].sum('Z').mean('time').plot(ax=ax[2],cbar_kwargs={"label": "", "aspect": 40})
    ax[2].set_title('kEpS')

    for n, axs in enumerate(ax):
        axs.text(-0.1, 1, string.ascii_lowercase[n], transform=axs.transAxes,
                size=20, weight='bold')
    plt.tight_layout()
    plt.savefig('./figs/myta_KE.png')
    #plt.show()

if 1:
    plt.clf()
    f, ax = plt.subplots(3, 1, figsize=(11,9) , sharey=True)
    
    ds1['kEpC'].sum('Z').mean('time').plot(ax=ax[0],cbar_kwargs={"label": "", "aspect": 40})
    ax[0].set_title('kEpC')
    ds1['Ep'].sum('Z').mean('time').plot(ax=ax[1],cbar_kwargs={"label": "", "aspect": 40})
    ax[1].set_title('Ep')
    myEbc.sum('Z').mean('time').plot(ax=ax[2],cbar_kwargs={"label": "", "aspect": 40})
    ax[2].set_title('Ebc')

    for n, axs in enumerate(ax):
        axs.text(-0.1, 1, string.ascii_lowercase[n], transform=axs.transAxes,
                size=20, weight='bold')
    plt.tight_layout()
    plt.savefig('./figs/myta_KE&PE&Ebc.png')
    #plt.show()


if 1:
    plt.clf()
    f, ax = plt.subplots(2, 2, figsize=(12,7.3) , sharey=True)
    ax=ax.flatten()

    ds1['uPbc'].sum('Z').mean('time').plot(ax=ax[0],cbar_kwargs={"label": "", "aspect": 40})
    ax[0].set_title('uPbc')
    ds1['vPbc'].sum('Z').mean('time').plot(ax=ax[1],cbar_kwargs={"label": "", "aspect": 40})
    ax[1].set_title('vPbc')

    ds2['SDIAG6'].mean('time').plot(ax=ax[2],cbar_kwargs={"label": "", "aspect": 40})
    ax[2].set_title('diag uPbc')
    ds2['SDIAG7'].mean('time').plot(ax=ax[3],cbar_kwargs={"label": "", "aspect": 40})
    ax[3].set_title('diag vPbc')

    for n, axs in enumerate(ax):
        axs.text(-0.1, 1, string.ascii_lowercase[n], transform=axs.transAxes,
                size=20, weight='bold')
    plt.tight_layout()
    plt.savefig('./figs/myta_uPbc&vPbc.png')
    #plt.show()


ds1['EpW']=xr.DataArray(ds1['Ep'].values,coords=[time1,z,yc,xg],dims=['time','Z','YC','XG'])
ds1['EpS']=xr.DataArray(ds1['Ep'].values,coords=[time1,z,yg,xc],dims=['time','Z','YG','XC'])

ds1['hkEpW']=(U0W*ds1['upW']+V0W*ds1['vpW'])*ds1['maskW']
ds1['hkEpS']=(U0S*ds1['upS']+V0S*ds1['vpS'])*ds1['maskS']

ds1['uEbc']=ds1['UVEL']*(ds1['kEpW']+ds1['hkEpW']+ds1['EpW'])*ds1['drF']*ds1['hFacW']*ds1['maskW']
ds1['vEbc']=ds1['VVEL']*(ds1['kEpS']+ds1['hkEpS']+ds1['EpS'])*ds1['drF']*ds1['hFacS']*ds1['maskS']


# calculate Conversion term no use differentiate -> use finite difference make coords fromXG to XC!!!!! W1.W2.ah0
ZW=(ds1.drF*ds1.hFacW).sum('Z')
ZS=(ds1.drF*ds1.hFacS).sum('Z')
ZC=(ds1.drF*ds1.hFacC).sum('Z')
print('ZW dims: ' +str(ZW.dims))
print('ZS dims: ' +str(ZS.dims))
print('ZC dims: ' +str(ZC.dims))

W1=grid.diff(ZW*U0W,'X', boundary='extrapolate')+grid.diff(ZS*V0S,'Y', boundary='extrapolate')
W2=grid.diff(U0W,'X', boundary='extrapolate')+grid.diff(V0S,'Y', boundary='extrapolate')
ds1['W']=-W1-W2*ZC
print(ds1['W'].dims)

upupW=ds1['upW']*ds1['upW']*ds1['drF']*ds1['hFacW']*ds1['maskW']
upvpS=ds1['upS']*ds1['vpS']*ds1['drF']*ds1['hFacS']*ds1['maskS']
upvpW=ds1['upW']*ds1['vpW']*ds1['drF']*ds1['hFacW']*ds1['maskW']
vpvpS=ds1['vpS']*ds1['vpS']*ds1['drF']*ds1['hFacS']*ds1['maskS']

conv1=ds1['rhoC']*g*ds1['W']*ds1['drF']*ds1['hFacC']/rhoNil
ah0=U0C*grid.diff(upupW,'X', boundary='extrapolate')+V0C*grid.diff(upvpW,'X', boundary='extrapolate')#+U0C*(upvS.differentiate('YG'))+V0C*(upvpS.differentiate('YG'))
print('con1 dims: '+str(conv1.dims))
print('ah0 dims: '+str(ah0.dims))

ds1['Conv']=conv1+ah0
print('ds1.Conv dims: '+str(ds1['Conv'].dims))

if 0:
    plt.clf()
    f, ax = plt.subplots(2, 1, figsize=(7.3,9) , sharey=True)
    
    ds1['hkEpW'].sum('Z').mean('time').plot(ax=ax[0],cbar_kwargs={"label": "", "aspect": 40})
    ax[0].set_title('hkEpW')
    ds1['hkEpS'].sum('Z').mean('time').plot(ax=ax[1],cbar_kwargs={"label": "", "aspect": 40})
    ax[1].set_title('hkEpS')

    for n, axs in enumerate(ax):
        axs.text(-0.1, 1, string.ascii_lowercase[n], transform=axs.transAxes,
                size=20, weight='bold')
    plt.tight_layout()
    plt.savefig('./figs/myta_hkEp.png')


if 1:
    plt.clf()
    f, ax = plt.subplots(2, 2, figsize=(12,7.3) , sharey=True)
    ax=ax.flatten()

    ds1['uEbc'].sum('Z').mean('time').plot(ax=ax[0],cbar_kwargs={"label": "", "aspect": 40})
    ax[0].set_title('uEbc')
    ds1['vEbc'].sum('Z').mean('time').plot(ax=ax[1],cbar_kwargs={"label": "", "aspect": 40})
    ax[1].set_title('vEbc')

    ds2['SDIAG8'].mean('time').plot(ax=ax[2],cbar_kwargs={"label": "", "aspect": 40})
    ax[2].set_title('diag uEbc')
    ds2['SDIAG9'].mean('time').plot(ax=ax[3],cmap='RdBu_r',cbar_kwargs={"label": "", "aspect": 40})
    ax[3].set_title('diag vEbc')


    for n, axs in enumerate(ax):
        axs.text(-0.1, 1, string.ascii_lowercase[n], transform=axs.transAxes,
                size=20, weight='bold')
    plt.tight_layout()
    plt.savefig('./figs/myta_uEbc&vEbc.png')


if 1:
    plt.clf()
    f, ax = plt.subplots(2, 1, figsize=(7.3,9) , sharey=True)
    
    ds2['SDIAG10'].mean('time').plot(ax=ax[0],vmin=-0.06,vmax=0.06,cmap='RdBu_r',cbar_kwargs={"label": "", "aspect": 40})
    ax[0].set_title('diag Conv')
    ds1['Conv'].sum('Z').mean('time').plot(ax=ax[1],vmin=-0.06,vmax=0.06,cmap='RdBu_r',cbar_kwargs={"label": "", "aspect": 40})
    ax[1].set_title('My Conv')

    for n, axs in enumerate(ax):
        axs.text(-0.1, 1, string.ascii_lowercase[n], transform=axs.transAxes,
                size=20, weight='bold')
    plt.tight_layout()
    plt.savefig('./figs/myta_Conv.png')
