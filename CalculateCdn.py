import numpy as np
import xarray as xr
import numpy.ma as ma
from uncertainties import unumpy as unp

import sys
import os

#sys.path.append('/home/bleins/ASET/ParamObsCDN')
from readlib import shebatower, shebatowergather, shebapam
from CDlib import *
from meteolib import *
import Interp_Ai_from_Aircraft


K0 = 273.15

def CalculateCdnSheeba(site=None,time_period='Entire',vmask=[],sub_CDN_skin=None,uncertainties=False,diag=False,theta_approx=False,LMO_approx=False,psi_stab='grachev',psi_unstab='businger-dyer'):
    """
    This function calculates the neutral drag coefficient at 10m (hourly and 10 day averaged) for the Sheba experiment. It return a xarray. If time_period == 'Aero_Summer', the function calculate the ice concentration from aircraft data from Perovich et al. (2002) (hourly and 10d average). Options are:
    - site amongst: Tower, Atlanta, Cleveland-Seattle-Maui, Baltimore, Florida
    - time_period: Entire or Aero_Summer
    - vmask: list providing the masking options to be used. Can be taken amonst the various screening described in Andreas et al. 2010 ('41a' to '42b' options) or be related to the uncertainty caused by the measurment random error propagated up to CDN ('err' option): if uncertainty is above a certain thershold
    - sub_CDN_skin: to substract the contribution of skin drag above water or ice (ice not codded yet).
    - uncertainties: include the uncertainties calculation. Option read measurement uncertainties from delta*.py. This option is CPU expensive as all the local partial derivative are computed.
    - diag: option to provide additional outputs (z0, LMO...)
    - theta_approx: logical for choosing or not the linear approximation for theta calculation
    - psi_stab and psi_unstab: option to choose which of the stability fonctions to use.

    """
    sites=['Atlanta','Cleveland-Seattle-Maui','Baltimore','Florida']

    ds = xr.Dataset()
    if site == 'Tower':
        xarr = shebatower()
        nlev = xarr.level.data.max()
        ### Height z ###################################
        z = xarr.z.data
        ### Ustar   #####################################
        ustar = xarr.ustar.data
        ### Temperature
        T = xarr.T.data+K0
        ### Pressure (hPa)
        #P = np.tile(xarr.Press.data*1.E2,(nlev,1)).transpose()
        P = np.tile(xarr.Press.data,(nlev,1)).transpose()
        #Do we compensate the pressure between different mast levels? Here it is constant.
        ### Relative humidity (%)
        rh = xarr.rh.data
        # Wind speed
        U_mod = xarr.ws.data
        #Sensible heat flux
        H = xarr.hs.data
        #Surf. temperature:
        #sigma = 5.67051E-8
        #eps = 0.99
        #Theta_s = (sigma * eps)**-0.25*(xarr.Lwu.data-(1-eps)*xarr.LWd.data)**0.25
        Ts = xarr.Tsfc.data+K0
        # Latent heat flux:
        #Warning, hl is considered constant between levels:
        LE = np.tile(xarr.hl.data,(nlev,1)).transpose()
        # Transform array into uarray with measurement uncertainties if needed:
        if uncertainties==True:
            import delta_ShebaTower as delta
            # Uncertainty package deal with std errors. Uncertainty are of B-type. As long as only bounds are provided for each measurment, a rectangular or uniform distribution of possible values is considered. As a result, the error standard deviation is delta/sqrt(3). Var=1/12(b-a)**2
            z = unp.uarray(z,np.tile(delta.z/np.sqrt(3),(np.shape(z))))
            ustar = unp.uarray(ustar,np.tile(delta.ustar/np.sqrt(3),(np.shape(ustar))))
            T = unp.uarray(T,np.tile(delta.T/np.sqrt(3),(np.shape(T))))
            P = unp.uarray(P,np.tile(delta.P/np.sqrt(3),(np.shape(P))))
            rh = unp.uarray(rh,np.tile(delta.rh/np.sqrt(3),(np.shape(rh))))
            U_mod = unp.uarray(U_mod,np.tile(delta.u/np.sqrt(3),(np.shape(U_mod))))
            H = unp.uarray(H,np.tile(delta.H/np.sqrt(3),(np.shape(H))))
            Ts = unp.uarray(Ts,np.tile(delta.Ts/np.sqrt(3),(np.shape(Ts))))
            LE =  unp.uarray(LE,np.abs(LE)*0.5/np.sqrt(3))

        q = Q(P=P,rh=rh,T=T)
        if theta_approx == True:
            theta = Theta(z=z,T=T)
        elif theta_approx == False:
            theta = Theta(T=T,P=P,q=q)
        thetav = Thetav(theta,q)
        rho = RHO(P=P,T=T,q=q)
        cp=1004
        # Create unp.uarray because the initial measured variables are wth and wq
        wth = H/(rho*cp)
        wth = unp.uarray(unp.nominal_values(wth),unp.std_devs(wth))
        lv = LV(T=T)
        # Create unp.uarray because the initial measured variables are wth and wq
        wq = LE/(rho*lv)
        wq = unp.uarray(unp.nominal_values(wq),unp.std_devs(wq))
        wthv = wth*(1.+0.51*q)+0.51*theta*wq
        #
    elif site[0] in sites:  
        xarr = shebapam(freq='1hour',sites=site)[0]
        nlev = 1
        ### Height z ###################################
        # WARNING, for 'Cleveland-Seattle-Maui'  xarr.Z_sonic_snowline is NaN for all points.
        #z = xarr.Z_sonic_snowline.data
        z = xarr.Z_sonic_meas.data
        ### Ustar   #####################################
        #tau = xarr.uflux.data
        #q = Q(P=xarr.P.data,rh=xarr.RH_ice.data,T=xarr.T.data+K0)
        # Check where is measured the pressure P?
        #rho = xarr.P.data*1.E2 / (287.058 *(xarr.T.data+K0))
        #rho = RHO(P=xarr.P.data,T=xarr.T.data+K0,q=q)
        #ustar = (tau / rho)**0.5
        #ustar = (xarr.u_v.data**2+xarr.u_w.data**2)**0.25
        ustar = (xarr.u_w**2)**0.25
        ### Temperature
        T = xarr.T.data+K0
        ### Pressure (hPa)
        P = xarr.P.data
        ### Relative humidity (%)
        rh = xarr.RH_ice.data
        ### Surface virtual temperature flux
        # Approximated as the sonic temperature flux (see Andreas et al. 2010)
        wthv = xarr.w_tc.data
        # Wind speed
        U_mod = xarr.spd.data
        # Transform array into uarray with measurement uncertainties if needed:
        if uncertainties==True:
            # Uncertainty package deal with std errors. Uncertainty are of B-type. As long as only bounds are provided for each measurment, a rectangular or uniform distribution of possible values is considered. As a result, the error standard deviation is delta/sqrt(3). Var=1/12(b-a)**2
            import delta_ShebaPAM as delta
            z = unp.uarray(z,np.tile(delta.z/np.sqrt(3),(np.size(z))))
            ustar = unp.uarray(ustar,np.tile(delta.ustar/np.sqrt(3),(np.size(ustar))))
            T = unp.uarray(T,np.tile(delta.T/np.sqrt(3),(np.size(T))))
            P = unp.uarray(P,np.tile(delta.P/np.sqrt(3),(np.size(P))))
            rh = unp.uarray(rh,np.tile(delta.rh/np.sqrt(3),(np.size(rh))))
            wthv = unp.uarray(wthv,np.tile(delta.wthv/np.sqrt(3),(np.size(wthv))))
            U_mod = unp.uarray(U_mod,np.tile(delta.u/np.sqrt(3),(np.size(U_mod))))
        ### Specific humidity (kg/kg)
        q = Q(P=P,rh=rh,T=T)
        ### Potential temperature
        if theta_approx == True:        
            theta = Theta(z=z,T=T)
        if theta_approx == False:
            theta = Theta(T=T,P=P,q=q)
        ### Virtual potential temperature
        thetav = Thetav(theta,q)
        #Surf. temperature:
        sigma = 5.67051E-8
        eps = 0.99
        Theta_s = (sigma * eps)**-0.25*(xarr.Rlw_out.data-(1-eps)*xarr.Rlw_in.data)**0.25
        # Set wth to wthv as no latent heat flux is measured at PAM stations. Only used for maksing data purposes. Anyway, it seems that wth is set to wthv in PAM stations following the providzed H values:
        wth = wthv
    else:
        sys.exit('A site have to be specified amongst \'Tower\' or one of the remote sites: '+str(sites))
    #######################################
    if LMO_approx == True:
        if site[0] in sites:
            sys.exit('LMO_approx is not possible for PAM stations as no humidity flux is available there.')
        L = LMOapprox(ustar=ustar, T=T, Q0=wthv, E0=wq)

    elif LMO_approx == False:
        L = LMO(ustar=ustar, thetav=thetav, Q0v=wthv)
    zeta = ZETA(z,L)
    psi = PSI(zeta, stab=psi_stab, unstab=psi_unstab)
    # In andreas et al. 2010, the effective wind is used. It is defined as the quad. sum of the mesured wind modulus plus either a thermal contribution for unstable conditions (Godfrey and Beljaars 1991) or a windless/meandering term for stable conditions (Jordan et al. 1999)
    # Here U_eff is calculated in two part in order to keep the traçaébility of U_mod in the final CDN (for uncertainties computation purposes)
    U_eff = UG(method='godfreybeljaars',u=U_mod,Q0v=wthv,h=600,beta=1.25,thetav=thetav,zeta=zeta)
    U_eff = UG(method='jordan',u=U_eff,zeta=zeta)
    #U_eff_unstable = UG(method='godfreybeljaars',u=U_mod,Q0v=wthv,h=600,beta=1.25,thetav=thetav,zeta=zeta)
    #U_eff_stable = UG(method='jordan',u=U_mod,zeta=zeta)
    #U_eff = np.where(zeta<0,U_eff_unstable,np.nan)
    #U_eff = np.where(zeta>0,U_eff_stable,np.nan)

    z0 = Z0(method='obs',u=U_eff,ustar=ustar,psi=psi[0],z=z)
    z0T = None
    z0Q = None
    cdn10 = CDN(z0=z0,z=10.)

    time_h = np.array(xarr.time.data, dtype='datetime64[s]')

########################
### Create time vector and indices of "10 days" averaged (10+10+8/10/11); 3 per month
### Which is the approach used in Andreas et al. 2010 but probably won't be used anymore
#########################
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
        t = 0
        # Divide month by 3
        for mb in np.arange(3):
            # The 2 first month averages are on 10 days...
            if mb < 2 :
                init=np.where((time_M==M - 12*int(M/13)) & (time_D==1+mb*10))[0][0]
                fint=np.where((time_M==M - 12*int(M/13)) & (time_D==1+mb*10+10))[0][0]
                ini.append(init)
                fin.append(fint)
                # ... while the third time period of the month varies from 8 to 11 days
            if mb == 2 :
                Y = time_Y[fint]
                init=fint
                fint=np.where((time_Y==Y) & (time_M==M - 12*int(M/13)))[0][-1]+1
                ini.append(init)
                fin.append(fint)
            time_10d.append(time_h[init]+np.timedelta64(time_h[fint]-time_h[init])/2)
        M +=1
#########################
    if time_period == 'Aero_Summer':
        # SHEBA aerodynamic summer following, e.g. Andreas et al. 2012
        time_period_ini = np.datetime64('1998-05-15T00:00')
        time_period_fin = np.datetime64('1998-09-14T23:59:59')
        #
        # Call of "Interp_Ai_Aircraft" function in order to read and interpolate ice fraction from aircraft and to ajust time vectors to the aicraft data time range.
        (Ai_period_h, Ai_period_10d, 
        time_10d_period, time_h_period, 
        ini10d, fin10d, 
        inih, finh) = Interp_Ai_from_Aircraft.Interp_Ai_Aircraft(time_h=time_h,
                                                                time_10d=time_10d,
                                                                time_period_ini=time_period_ini,
                                                                time_period_fin=time_period_fin)
        if sub_CDN_skin == 'water':
            z0w =  Z0(method='coare2.5',ustar=ustar,alpha=0.011,T=T)
            cdn10w = CDN(z0=z0w,z=10.)
            #
            if site == 'Tower':
                cdn10[inih:finh] -= (1-np.tile(Ai_period_h,(nlev,1)).transpose())*cdn10w[inih:finh]
            else:
                cdn10[inih:finh] -= (1-Ai_period_h)*cdn10w[inih:finh]
        

    if time_period == 'Entire':
        time_period_ini = xarr.time.data[0]
        time_period_fin = xarr.time.data[-1]
        #
        ini10d = np.where(time_10d>=time_period_ini)[0][0]
        fin10d = np.where(time_10d<=time_period_fin)[0][-1]+1
        inih = np.where(time_h>=time_period_ini)[0][0]
        finh = np.where(time_h<=time_period_fin)[0][-1]+1

    ##################################################
    # Mask creation
    ##################################################
    mask = np.zeros(ustar.shape)
    # Mask criteria 41a to 42b come from Equations 41a to 42b in Andreas et al. 2010
    if '41a' in vmask: 
        mask = np.where(ustar<=0.,1,mask)
    if '41b' in vmask:
        mask = np.where(np.abs(wth)<=0.005,1,mask)
    if '41c' in vmask:
        sys.exit('Mask option 41c not tested yet, see Andreas et al. 2010')
    if '41d' in vmask:
        mask = np.where(np.abs(unp.nominal_values(Theta_s)-unp.nominal_values(T))<=0.5,1,mask)
    if '41e' in vmask:
        sys.exit('Mask option 41e not tested yet, see Andreas et al. 2010')
    if '42a' in vmask:
        mask = np.where(z0>=0.1,1,mask)
        if z0T != None:
            mask = np.where(z0T>=0.1,1,mask)
        if z0Q != None:
            mask = np.where(z0Q>=0.1,1,mask)
    if '42b' in vmask:
        sys.exit('Mask option 42b not tested yet, see Andreas et al. 2010')
        if z0T != None:
            mask = np.where(z0T<=7.E-8,1,mask)
        if z0Q != None:
            mask = np.where(z0Q<=7.E-8,1,mask)
    if 'QC' in vmask:
        if site == 'Tower':
            mask = np.where(xarr.fl.data==1,1,mask)
        else:
            # Warning, QC check for PAM stations not understood yet, recheck online doc!
            mask = np.where(xarr.samples_sonic.data<=99.5,1,mask)
    # Mask from uncertainty criteria:
    if 'err' in vmask:
        mask = np.where(unp.std_devs(cdn10)>=2*unp.nominal_values(cdn10),1,mask)

    L = ma.masked_where(mask==1,L)
    zeta = ma.masked_where(mask==1,zeta)
    if site == 'Tower':
        psi = ma.masked_where(np.tile(mask,(2,1,1))==1,psi)
    else:
        psi = ma.masked_where(np.tile(mask,(2,1))==1,psi)
    cdn10 = ma.masked_where(mask==1,cdn10)
    ##################################################
    if site == 'Tower':
    # Median CDN have to be computed after masking runs for each level
        cdn10_med = np.ma.median(unp.nominal_values(cdn10),axis=1)
            
    #########################################################
    # Averaging CDN on 10 days period: 10+10+(8 or 10 or 11)
    #########################################################
    if site == 'Tower':
        # Calculate CDN median 10 days averaged and corresponding time vector 
        cdn_med_10d = []
        cdn_med_10d_iqr = []
        cdn_med_10d_std = []
        # Start averaging 1st of november (first entire month). Expe started 29th of october... 
        for j in np.arange(np.size(ini)):
            cdn10_med_temp = cdn10_med[ini[j]:fin[j]]
            cdn_med_10d.append(np.nanmean(cdn10_med_temp))
            cdn_med_10d_iqr.append(np.array([np.nanpercentile(cdn10_med_temp.compressed(),25),np.nanpercentile(cdn10_med_temp.compressed(),75)]))
            cdn_med_10d_std.append(np.nanstd(cdn10_med_temp.compressed()))

        cdn_med_10d_period = np.array(cdn_med_10d[ini10d:fin10d])
        cdn_med_10d_period_iqr = np.array(cdn_med_10d_iqr[:][ini10d:fin10d])
        cdn_med_10d_period_std = np.array(cdn_med_10d_std[:][ini10d:fin10d])

    else:
        cdn_10d = []
        cdn_10d_iqr = []
        cdn_10d_std = []
        # Start averaging 1st of november (first entire month). Expe started 29th of october... 
        for j in np.arange(np.size(ini)):
            #Warning: some 10 days periods may have 100% of masked values or 100% nan. Return NaN in such cases.
            cdn10_temp = cdn10[ini[j]:fin[j]]
            test='ok'
            if (False in unp.isnan(cdn10_temp))==False:
                test='nan'
            if np.size(cdn10_temp.mask)!=1:
                if ((False in cdn10_temp.mask)) == False :
                    test='nan'
            #
            if test=='ok':
                cdn_10d.append(cdn10_temp[~unp.isnan(cdn10_temp)].mean())
                cdn_10d_iqr.append(np.array([np.percentile(cdn10_temp[~unp.isnan(cdn10_temp)],25),np.percentile(cdn10_temp[~unp.isnan(cdn10_temp)],75)]))
                cdn_10d_std.append(unp.sqrt(np.mean(np.array(cdn10_temp[~unp.isnan(cdn10_temp)]-cdn10_temp[~unp.isnan(cdn10_temp)].mean())**2)))

            if test=='nan':
                cdn_10d.append(np.nan)
                cdn_10d_iqr.append(np.array([np.nan,np.nan]))
                cdn_10d_std.append(np.nan)

        cdn_10d_period = np.array(cdn_10d[ini10d:fin10d])
        cdn_10d_period_iqr = np.array(cdn_10d_iqr[:][ini10d:fin10d])
        cdn_10d_period_std = np.array(cdn_10d_std[:][ini10d:fin10d])

    ############################################################
    ### Reduce hourly vectors on the disired time period
    ############################################################
    if site == 'Tower':
        cdn_med_h_period = np.ma.array(cdn10_med[inih:finh])
        cdn_h_period = np.ma.array(cdn10[inih:finh,:])
    else:
        cdn_h_period = np.ma.array(cdn10[inih:finh])
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
        
        array = xr.DataArray(cdn_med_10d_period_iqr,dims=('time_10d','nq'),attrs={'long_name': 'CDN10 5 levels tower median and 10 days IQR'})
        ds['cdn_med_10d_iqr'] = array
        nq=np.array([25,75])
        ds=ds.assign_coords(nq=nq)
  
        array = xr.DataArray(cdn_med_10d_period_std,dims=('time_10d'),attrs={'long_name': 'CDN10 5 levels tower median and 10 days STD'})
        ds['cdn_med_10d_std'] = array

        # In DataArrays, masked data is represented with NaN values
        array = xr.DataArray(cdn_med_h_period,dims=('time_h'),attrs={'long_name': 'CDN10 5 levels tower median (hourly)'})
        ds['cdn_med_h'] = array
    
        array = xr.DataArray(cdn_h_period,dims=('time_h','height'),attrs={'long_name': 'CDN10 at available tower levels (hourly)'})
        ds['cdn_h'] = array
        height = np.arange(nlev)+1
        ds=ds.assign_coords(height=height)

    else:
        array = xr.DataArray(cdn_10d_period,dims=('time_10d'),attrs={'long_name': 'CDN10 10 days mean'})
        ds['cdn_10d'] = array
        
        array = xr.DataArray(cdn_10d_period_iqr,dims=('time_10d','nq'),attrs={'long_name': 'CDN10 10 days IQR'})
        ds['cdn_10d_iqr'] = array
        nq=np.array([25,75])
        ds=ds.assign_coords(nq=nq)
        
        array = xr.DataArray(cdn_10d_period_std,dims=('time_10d'),attrs={'long_name': 'CDN10 10 days STD'})
        ds['cdn_10d_std'] = array
        
        array = xr.DataArray(cdn_h_period,dims=('time_h',),attrs={'long_name': 'CDN10 at available tower levels (hourly)'})
        ds['cdn_h'] = array

    if time_period == 'Aero_Summer':
        array = xr.DataArray(Ai_period_10d,dims=('time_10d'),attrs={'long_name': '10 days averaged ice fraction from Perrovich et al. 2002'})
        ds['Ai_10d'] = array
        #Hourly
        array = xr.DataArray(Ai_period_h,dims=('time_h'),attrs={'long_name': 'Hourly interpolated ice fraction from Perrovich et al. 2002'})
        ds['Ai_h'] = array
       
        ds=ds.assign_coords(time_10d=time_10d_period)
    elif time_period == 'Entire':
        ds=ds.assign_coords(time_10d=time_10d)

    ds=ds.assign_coords(time_h=time_vect)

    if diag == True:
        if site == 'Tower':
            return (ds, 
                    cdn_h_period,
                    Ai_period_h,

                    ma.masked_where(mask==1,z)[inih:finh],
                    ma.masked_where(mask==1,ustar)[inih:finh],
                    ma.masked_where(mask==1,T)[inih:finh],
                    ma.masked_where(mask==1,P)[inih:finh],
                    ma.masked_where(mask==1,rh)[inih:finh],
                    ma.masked_where(mask==1,U_mod)[inih:finh],
                    ma.masked_where(mask==1,U_eff)[inih:finh],
                    ma.masked_where(mask==1,z0)[inih:finh],
                    ma.masked_where(mask==1,q)[inih:finh],
                    ma.masked_where(mask==1,theta)[inih:finh],
                    ma.masked_where(mask==1,thetav)[inih:finh],
                    ma.masked_where(mask==1,wthv)[inih:finh],
                    #
                    ma.masked_where(mask==1,wth)[inih:finh],
                    ma.masked_where(mask==1,wq)[inih:finh],
                    ma.masked_where(mask==1,H)[inih:finh],
                    #

                    L[inih:finh],
                    zeta[inih:finh],
                    psi[0][inih:finh],
                    )
        else:
            return (ds, 
                    cdn_h_period,
                    Ai_period_h,

                    ma.masked_where(mask==1,z)[inih:finh],
                    ma.masked_where(mask==1,ustar)[inih:finh],
                    ma.masked_where(mask==1,T)[inih:finh],
                    ma.masked_where(mask==1,P)[inih:finh],
                    ma.masked_where(mask==1,rh)[inih:finh],
                    ma.masked_where(mask==1,U_mod)[inih:finh],
                    ma.masked_where(mask==1,U_eff)[inih:finh],
                    ma.masked_where(mask==1,z0)[inih:finh],
                    ma.masked_where(mask==1,q)[inih:finh],
                    ma.masked_where(mask==1,theta)[inih:finh],
                    ma.masked_where(mask==1,thetav)[inih:finh],
                    #
                    ma.masked_where(mask==1,wthv)[inih:finh],
                    #

                    L[inih:finh],
                    zeta[inih:finh],
                    psi[0][inih:finh],
                    )
    elif diag== False:
        return ds

