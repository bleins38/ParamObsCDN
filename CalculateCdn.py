import numpy as np
import xarray as xr
import numpy.ma as ma
import sys
import os
sys.path.append('/home/bleins/ASET/TurbObsLibrary')
sys.path.append('/home/bleins/ASET/meteolib')
sys.path.append('/home/bleins/ASET/CDlib')
sys.path.append('/home/bleins/ASET/SCRIPT')

from readlib import shebatower, shebatowergather, shebapam
from CDlib import *
from meteolib import *
import Interp_Ai_from_Aircraft


K0 = 273.15
sites=['Atlanta','Cleveland-Seattle-Maui','Baltimore','Florida']

def CalculateCdnSheeba(site=None,time_period=None,vmask=None,sub_CDN_skin=None):
    ds = xr.Dataset()
    if site == 'Tower':
        xarr = shebatower()
        nlev = xarr.level.data.max()
        z = xarr.z.data
        #
        #P = np.tile(xarr.Press.data*1.E2,(nlev,1)).transpose()
        P = np.tile(xarr.Press.data,(nlev,1)).transpose()
        #Do we compensate the pressure between different mast levels? Here it is constant.
        #rho = P/(287.058 *(xarr.T.data+K0))
        rho = RHO(P=P,T=xarr.T.data+K0,q=xarr.q.data*1.E-3)
        #
        ustar = xarr.ustar.data
        #
        theta = Theta(z=z,T=xarr.T.data+K0)
        thetav = Thetav(theta,xarr.q.data*1.E-3)

        wth = xarr.hs.data/(rho*cp)
        lv = LV(T=xarr.T.data+K0)

        #Warning, hl is considered constant between levels:
        LE = np.tile(xarr.hl.data,(nlev,1)).transpose()
        wq = LE/(rho*lv)
        wthv = wth*(1.+0.51*xarr.q.data*1.E-3)+0.51*theta*wq
        #
        U_mod = xarr.ws.data
        #
        #Surf. temperature:
        #sigma = 5.67051E-8
        #eps = 0.99
        #Theta_s = (sigma * eps)**-0.25*(xarr.Lwu.data-(1-eps)*xarr.LWd.data)**0.25
        Theta_s = xarr.Tsfc.data

    elif site[0] in sites :  
        xarr = shebapam(freq='1hour',sites=site)[0]
        nlev = 1
        # WARNING, for 'Cleveland-Seattle-Maui'  xarr.Z_sonic_snowline is NaN for all points.
        #z = xarr.Z_sonic_snowline.data
        z = xarr.Z_sonic_meas.data
        #
        tau = xarr.uflux.data
        q = Q(P=xarr.P.data*1.E2,rh=xarr.RH_ice.data,T=xarr.T.data+K0)
        # Check where is measured the pressure P?
        #rho = xarr.P.data*1.E2 / (287.058 *(xarr.T.data+K0))
        rho = RHO(P=xarr.P.data*1.E2,T=xarr.T.data+K0,q=q)
        ustar = (tau / rho)**0.5
        #ustar = (xarr.u_v.data**2+xarr.u_w.data**2)**0.25
        #
        theta = Theta(z=z,T=xarr.T.data+K0)
        thetav = Thetav(theta,q)
        wthv = xarr.w_tc.data
        #
        U_mod = xarr.spd.data
        #Surf. temperature:
        sigma = 5.67051E-8
        eps = 0.99
        Theta_s = (sigma * eps)**-0.25*(xarr.Rlw_out.data-(1-eps)*xarr.Rlw_in.data)**0.25

    else:
     sys.exit('A site have to be specified amongst \'Tower\' or one of the remote sites: '+str(sites))
    if time_period == 'Aero_Summer':
        # SHEBA aerodynamic summer following, e.g. Andreas et al. 2012
        time_period_ini = np.datetime64('1998-05-15T00:00')
        time_period_fin = np.datetime64('1998-09-14T23:59:59')
    elif time_period == 'Entire':
        time_period_ini = xarr.time.data[0]
        time_period_fin = xarr.time.data[-1]

    #######################################
    LMO_func = LMO(ustar,thetav,Q0v=wthv)


    zeta = z/LMO_func

    psi_func = PSI(z,LMO_func,stab='grachev',unstab='businger-dyer')
    psi_func = np.array(psi_func)

    # In andreas et al. 2010, they used the effective wind defined as the sum of the mesured wind modulus plus either a thermal contribution for unstable conditions or a windless/meandering term for stable conditions
    #U_mod = xarr.ws.data
    # Convective velocity scale (Godfrey and Beljaars 1991)
    #betag = 1.25
    #zi = 600
    #k = 0.4
    #wstar = ustar * (- zi /(k * LMO_func))**(1./3)
    #
    #U_eff = np.where(zeta<0,(U_mod**2+betag**2*wstar**2)**0.5,U_mod)
    #U_eff = np.where(zeta>0,U_mod+0.5/np.cosh(U_mod),U_eff)
    U_eff = np.where(zeta<0,UG(method='godfreybeljaars',u=U_mod,Q0v=wthv,h=600,thetav=thetav),U_mod)
    U_eff = np.where(zeta>0,UG(method='jordan',u=U_mod),U_eff)

########################
### Create time vector and incices of "10 days" averaged (10+10+8/10/11); 3 per month
#########################
    time_h = np.array(xarr.time.data, dtype='datetime64[s]')
    #
    time_D = np.array([time_h[i].astype(object).day for i in np.arange(time_h.shape[0])])
    time_M = np.array([time_h[i].astype(object).month for i in np.arange(time_h.shape[0])])
    time_Y = np.array([time_h[i].astype(object).year for i in np.arange(time_h.shape[0])])
    #
    expe_ini = time_h[0]
    expe_fin = time_h[-1]
    # Total expe month number
    Nmonth = np.timedelta64(np.array(expe_fin, dtype='datetime64[M]')-np.array(expe_ini, dtype='datetime64[M]')).astype(int)
    time_10d = []
    # Start averaging 1st of november (first entire month). Expe started 29th of october... 
    ini=[]
    fin=[]
    for M in np.arange(time_h[0].astype(object).month+1,time_h[0].astype(object).month+Nmonth-1):
        #print(M)
        #print(M - 12*int(M/13))
        t = 0
        # Divide month by 3
        for mb in np.arange(3):
            # The 2 first month averages are on 10 days...
            if mb < 2 :
                init=np.where((time_M==M - 12*int(M/13)) & (time_D==1+mb*10))[0][0]
                fint=np.where((time_M==M - 12*int(M/13)) & (time_D==1+mb*10+10))[0][0]
                ini.append(init)
                fin.append(fint)
                #print(time_h[ini])
                #print(time_h[fin])
                # ... will the third time period of the month varies from 8 to 11 days
            if mb == 2 :
                Y = time_Y[fint]
                init=fint
                fint=np.where((time_Y==Y) & (time_M==M - 12*int(M/13)))[0][-1]+1
                ini.append(init)
                fin.append(fint)
                #print(time_h[ini])
                #print(time_h[fin])
            #print(cdn10_func.shape)
            #print(ini,fin)
            time_10d.append(time_h[init]+np.timedelta64(time_h[fint]-time_h[init])/2)
        M +=1

    if time_period == 'Aero_Summer':
        # Call of "Interp_Ai_Aircraft" function in order to read and interpolate ice fraction from aircraft and to ajust time vectors to the aicraft data time range.
        (Ai_period_h, Ai_period_10d, 
        time_10d_period, time_h_period, 
        ini10d, fin10d, 
        inih, finh) = Interp_Ai_from_Aircraft.Interp_Ai_Aircraft(time_h=time_h,
                                                                time_10d=time_10d,
                                                                time_period_ini=time_period_ini,
                                                                time_period_fin=time_period_fin)
    if time_period == 'Entire':
        ini10d = np.where(time_10d>=time_period_ini)[0][0]
        fin10d = np.where(time_10d<=time_period_fin)[0][-1]+1
        inih = np.where(time_h>=time_period_ini)[0][0]
        finh = np.where(time_h<=time_period_fin)[0][-1]+1

    z0_func = Z0(method='obs',u=U_eff,ustar=ustar,psi=psi_func[0],z=z)
    cdn10_func = CDN(z0=z0_func,z=10.)
    if time_period == 'Aero_Summer':
        if sub_CDN_skin == 'water':
            z0w =  Z0(method='coare2.5',ustar=ustar,alpha=0.011,T=xarr.T.data+K0)
            cdn10w = CDN(z0=z0w,z=10.)
            #
            if site == 'Tower':
                cdn10_func[inih:finh] -= (1-np.tile(Ai_period_h,(nlev,1)).transpose())*cdn10w[inih:finh]
            else:
                cdn10_func[inih:finh] -= (1-Ai_period_h)*cdn10w[inih:finh]
    ##################################################
    # Mask creation
    ##################################################
    if vmask == 'nomask':
        mask = np.zeros(ustar.shape)
    if vmask == 'QC_flag':
        mask = np.zeros(ustar.shape)
        mask = np.where(xarr.fl.data==1,1,mask)
    if vmask == 'Andreas2010_41ab':
        mask = np.zeros(ustar.shape)
        mask = np.where(ustar<=0.,1,mask)
        mask = np.where(np.abs(wth)<=0.005,1,mask)
    if vmask == 'Andreas2010_42a':
        mask = np.zeros(ustar.shape)
        mask = np.where(z0_func>=0.1,1,mask)
    if vmask == 'Andreas2010_41ab_42a':
        mask = np.zeros(ustar.shape)
        mask = np.where(ustar<=0.,1,mask)
        mask = np.where(np.abs(wth)<=0.005,1,mask)
        mask = np.where(z0_func>=0.1,1,mask)
    if vmask == 'QC_flag_Andreas2010_41ab_42a':
        mask = np.zeros(ustar.shape)
        mask = np.where(xarr.fl.data==1,1,mask)
        mask = np.where(ustar<=0.,1,mask)
        mask = np.where(np.abs(wth)<=0.005,1,mask)
        mask = np.where(z0_func>=0.1,1,mask)
    if vmask == 'QC_flag_Andreas2010_42a':
        mask = np.zeros(ustar.shape)
        mask = np.where(xarr.fl.data==1,1,mask)
        mask = np.where(z0_func>=0.1,1,mask)
    if vmask == 'QC_flag_Andreas2010_41ab':
        mask = np.zeros(ustar.shape)
        mask = np.where(xarr.fl.data==1,1,mask)
        mask = np.where(ustar<=0.,1,mask)
        mask = np.where(np.abs(wth)<=0.005,1,mask)
    #Mask for PAM stations:
    if vmask == 'PAM_samples_sonic':
        mask = np.zeros(ustar.shape)
        mask = np.where(xarr.samples_sonic.data<=99.5,1,mask)
    #Mask criteria from Andreas et al. 2010:
    #Eq. 4.1 a,b,c,d,e
    ###mask = np.where(ustar<=0.,1,mask)
    ###mask = np.where(np.abs(wth)<=0.005,1,mask)
    #mask = np.where(np.abs(wq)<=2.5E-7,1,mask)
    #mask = np.where(np.tile(Theta_s - K0,(nlev,1)).transpose() - xarr.T.data<=0.5,1,mask)
    #Eq. 4.2 a, b
    #mask = np.where(z0_func>=0.1,1,mask)
    #mask = np.where(z0_func<=7.E-8,1,mask)

    LMO_func = ma.masked_where(mask==1,LMO_func)
    zeta = ma.masked_where(mask==1,zeta)
    if site == 'Tower':
        psi_func = ma.masked_where(np.tile(mask,(2,1,1))==1,psi_func)
    else:
        psi_func = ma.masked_where(np.tile(mask,(2,1))==1,psi_func)
    cdn10_func = ma.masked_where(mask==1,cdn10_func)
    ##################################################
    if site == 'Tower':
    # Median CDN have to be computed after masking runs for each level
        cdn10_med = np.ma.median(cdn10_func,axis=1)

    #########################################################
    # Averaging CDN on 10 days period: 10+10+(8 or 10 or 11)
    #########################################################
    if site == 'Tower':
        # Calculate CDN median 10 days averaged and corresponding time vector 
        cdn_med_10d = []
        cdn_med_10d_iqr = []
        cdn_med_10d_std = []

    else:
        cdn_10d = []
        cdn_10d_iqr = []
        cdn_10d_std = []
    # Start averaging 1st of november (first entire month). Expe started 29th of october... 
    for j in np.arange(np.size(ini)):
        if site == 'Tower':
            cdn_med_10d.append(np.nanmean(cdn10_med[ini[j]:fin[j]]))
            cdn_med_10d_iqr.append(np.array([np.nanpercentile(cdn10_med[ini[j]:fin[j]].compressed(),25),np.nanpercentile(cdn10_med[ini[j]:fin[j]].compressed(),75)]))
            cdn_med_10d_std.append(np.nanstd(cdn10_med[ini[j]:fin[j]].compressed()))
        else:
            #Warning: some 10 days periods may have 100% of masked values. Return NaN in such cases.
            if vmask != 'nomask':
                if (False in cdn10_func[ini[j]:fin[j]].mask) == True :
                    cdn_10d.append(np.nanmean(cdn10_func[ini[j]:fin[j]]))
                    cdn_10d_iqr.append(np.array([np.nanpercentile(cdn10_func[ini[j]:fin[j]].compressed(),25),np.nanpercentile(cdn10_func[ini[j]:fin[j]].compressed(),75)]))
                    cdn_10d_std.append(np.nanstd(cdn10_func[ini[j]:fin[j]].compressed()))
                if (False in cdn10_func[ini[j]:fin[j]].mask) == False :
                    cdn_10d.append(np.nan)
                    cdn_10d_iqr.append(np.array([np.nan,np.nan]))
                    cdn_10d_std.append(np.nan)
            else:
                    cdn_10d.append(np.nanmean(cdn10_func[ini[j]:fin[j]]))
                    cdn_10d_iqr.append(np.array([np.nanpercentile(cdn10_func[ini[j]:fin[j]].compressed(),25),np.nanpercentile(cdn10_func[ini[j]:fin[j]].compressed(),75)]))
                    cdn_10d_std.append(np.nanstd(cdn10_func[ini[j]:fin[j]].compressed()))


    # From tower data:
    # 10days average
    #
    if site == 'Tower':
        cdn_med_10d_period = np.array(cdn_med_10d[ini10d:fin10d])
        cdn_med_10d_period_iqr = np.array(cdn_med_10d_iqr[:][ini10d:fin10d])
        cdn_med_10d_period_std = np.array(cdn_med_10d_std[:][ini10d:fin10d])
    else:
        cdn_10d_period = np.array(cdn_10d[ini10d:fin10d])
        cdn_10d_period_iqr = np.array(cdn_10d_iqr[:][ini10d:fin10d])
        cdn_10d_period_std = np.array(cdn_10d_std[:][ini10d:fin10d])

    # hourly
    #
    if site == 'Tower':
        cdn_med_h_period = np.ma.array(cdn10_med[inih:finh])
        cdn_h_period = np.ma.array(cdn10_func[inih:finh,:])
    else:
        cdn_h_period = np.ma.array(cdn10_func[inih:finh])
    #height_summer = np.array(z[ini:fin,:])
    time_vect = time_h[inih:finh]

    ###########################################################
    ###########################################################
    ###########################################################
    ### Fill output xarray
    #10 days averaged data
    if site == 'Tower':
        array = xr.DataArray(cdn_med_10d_period,dims=('time_10d'),attrs={'long_name': 'CDN10 5 levels tower median and 10 days mean'})
        ds['cdn_med_10d'] = array
        #
        array = xr.DataArray(cdn_med_10d_period_iqr,dims=('time_10d','nq'),attrs={'long_name': 'CDN10 5 levels tower median and 10 days IQR'})
        ds['cdn_med_10d_iqr'] = array
        #
        array = xr.DataArray(cdn_med_10d_period_std,dims=('time_10d'),attrs={'long_name': 'CDN10 5 levels tower median and 10 days STD'})
        ds['cdn_med_10d_std'] = array

        #
        nq=np.array([25,75])
        ds=ds.assign_coords(nq=nq)
    else:
        array = xr.DataArray(cdn_10d_period,dims=('time_10d'),attrs={'long_name': 'CDN10 10 days mean'})
        ds['cdn_10d'] = array
        #
        array = xr.DataArray(cdn_10d_period_iqr,dims=('time_10d','nq'),attrs={'long_name': 'CDN10 10 days IQR'})
        ds['cdn_10d_iqr'] = array
        #
        array = xr.DataArray(cdn_10d_period_std,dims=('time_10d'),attrs={'long_name': 'CDN10 10 days STD'})
        ds['cdn_10d_std'] = array
        #
        nq=np.array([25,75])
        ds=ds.assign_coords(nq=nq)
    ##
    if time_period == 'Aero_Summer':
        array = xr.DataArray(Ai_period_10d,dims=('time_10d'),attrs={'long_name': '10 days averaged ice fraction from Perrovich et al. 2002'})
        ds['Ai_10d'] = array
        #Hourly
        array = xr.DataArray(Ai_period_h,dims=('time_h'),attrs={'long_name': 'Hourly interpolated ice fraction from Perrovich et al. 2002'})
        ds['Ai_h'] = array
        #
        ds=ds.assign_coords(time_10d=time_10d_period)
    elif time_period == 'Entire':
        ds=ds.assign_coords(time_10d=time_10d)

    if site == 'Tower':
    # In DataArrays, masked data is represented with NaN values
        array = xr.DataArray(cdn_med_h_period,dims=('time_h'),attrs={'long_name': 'CDN10 5 levels tower median (hourly)'})
        ds['cdn_med_h'] = array
    #
        array = xr.DataArray(cdn_h_period,dims=('time_h','height'),attrs={'long_name': 'CDN10 at available tower levels (hourly)'})
        ds['cdn_h'] = array
        height = np.arange(nlev)+1
        ds=ds.assign_coords(height=height)
    else:
        array = xr.DataArray(cdn_h_period,dims=('time_h',),attrs={'long_name': 'CDN10 at available tower levels (hourly)'})
        ds['cdn_h'] = array
    ds=ds.assign_coords(time_h=time_vect)

    #
    return ds
