import numpy as np
import sys

sys.path.append('/home/bleins/ASET/TurbObsLibrary')
import readlib

def Interp_Ai_Aircraft(time_h=None,time_10d=None,time_period_ini=None,time_period_fin=None):
    ###################################################################
    # Read aircraft data (ice concentration) from Perovich et al. 2002
    ###################################################################
    xa_aircraft = readlib.shebaaircraft()
    ###########
    #Create time vector (10 days centered period) corresponding to the time period where the aircraft data are available
    ini10d_period = np.max([np.where(time_10d>=xa_aircraft.date.data[0])[0][0]-1,
                            np.where(time_10d>=time_period_ini)[0][0]])
    fin10d_period = np.min([np.where(time_10d<=xa_aircraft.date.data[-1])[0][-1]+1,
                            np.where(time_10d<=time_period_fin)[0][-1]+1])
    time_10d_period = time_10d[ini10d_period:fin10d_period]
    # Convert time vectors (of both aircraft data and 10 days centered period) into float in order to interpolate
    time_10d_period_float = np.array([np.array([time_10d_period[i]], dtype='datetime64[s]').astype("float")[0] for i in np.arange(len(time_10d_period))])
    time_aircraft_float = np.array(xa_aircraft.date.data[:], dtype='datetime64[s]').astype("float")

    # Interpolate (linear) sea ice concentration between aircraft data times
    Ai_period_10d = np.interp(time_10d_period_float,time_aircraft_float,xa_aircraft.Ai)

    ###########
    #Create time vector (hourly) corresponding to the time period where the aircraft data are available
    inih_period = np.max([np.where(time_h>=xa_aircraft.date.data[0])[0][0],#-1
                            np.where(time_h>=time_period_ini)[0][0]])
    finh_period = np.min([np.where(time_h<=xa_aircraft.date.data[-1])[0][-1]+1,
                             np.where(time_h<=time_period_fin)[0][-1]+1])
    time_h_period = time_h[inih_period:finh_period]
    # Convert time vectors (of both aircraft data and hourly corresponding vector) into float in order to interpolate
    time_h_aircraft_float = np.array([np.array([time_h_period[i]], dtype='datetime64[s]').astype("float")[0] for i in np.arange(len(time_h_period))])

    # Interpolate (linear) sea ice concentration between aircraft data times
    Ai_period_h = np.interp(time_h_aircraft_float,time_aircraft_float,xa_aircraft.Ai)
    #
    ini10d = ini10d_period
    fin10d = fin10d_period
    inih = inih_period
    finh = finh_period

    return Ai_period_h, Ai_period_10d, time_10d_period, time_h_period, ini10d, fin10d, inih, finh
