#############################################
# Measurement uncertainties for Tower :
#############################################
# PAM website:
# https://www.eol.ucar.edu/content/sheba-isff-flux-pam-project-report#DataArchive

# Altitude/Measurement levels:
# "heights were obtained using the snow depth measurements and occasional manual height measurements" and "a stake to provide snow depth measurements" in Persson et al. (2002). No uncertainty is provided. Here we assumed that the uncertainty may com from:
# (i) the measure itself (1cm);
# (ii) from the fact that the measure is local and the ground may be etherogeneous (3cm) and 
# (iii) "During JD 514-556, the snow depth has been linearly adjusted for a settling of the stake of 21 cm." We assume this correction may lead to an additional uncertainty of 0.5cm (21cm on 42 days...) We assume no uncertainty from the time sampling of this measurement (1/day or less in Persson et al. 2002)
z=0.01+0.03+0.005

# ustar: 
# +- 0.05m/s in Persson et al. 2002 (random errors hourly)
# Applied. Tech. Inc. provides a wind speed accuacy of +- 0.01m/s (for nowaday sensor... K-type spec. have changed since Sheba?)
# https://www.apptech.com/products/ultrasonic-anemometers/specifications/
ustar=0.05

# Air temperature:
# +- 0.05 deg. C in Persson et al. 2002 (random error hourly)
T=0.05

# Pressure:
# "Hourly averages of atmospheric pressure, calculated from 5-minute means, were obtained directly from the Portable Automated Mesonet (PAM) station named ‘‘Florida’’"
# 0.3mb reduced to 0.1mb in PAM website (error ?)
P=0.1 # (hPa)

# Relative humidity:
# +- 1% in Persson et al. 2002 (random error hourly)
rh=1 #(%)

#Sensible heat flux:
# +- 4.1 W.m-2 in Persson et al. 2002 (random errors hourly)
H=4.1 #(W.m-2)

# Surface temperature
# +- 0.6 deg. C from Persson et al. 2002  (random Errors hourly)
Ts= 0.6 #(deg. C)

# Wind speed:
# Typical from R3 Gill, to be ajusted
u=0.1
# Apllied Tech. Inc gives for K-type sonic 0.01m/s
#u=0.01
