#############################################
# Measurement uncertainties for PAM stations:
#############################################
# PAM website:
# https://www.eol.ucar.edu/content/sheba-isff-flux-pam-project-report#DataArchive
# Altitude/Measurement levels:
# measured manually during station maintenance visits (see PAM website). No uncertainty is provided. Here we assumed that the uncertainty may com from:
# (i) the measure itself (1cm); 
# (ii) from the fact that the measure is local and the ground may be etherogeneous (3cm) and 
# (iii) from the time subsampling of the measurement (roughly every 15 days)
z=0.01+0.03+0.05

# ustar: 
# +- 0.05m/s in Persson et al. 2002 (random error hourly)
ustar=0.05

# Air temperature:
# +- 0.05 deg. C in Persson et al. 2002 (random error hourly)
T=0.05

# Pressure:
# 0.3mb reduced to 0.1mb in PAM website (random error hourly)
P=0.1 # (hPa)

# Relative humidity:
# +- 1% in Persson et al. 2002 (random error hourly)
rh=1 #(%)

# Surface virtual temperature
# Sonic specification to be found... warning, several sonic sensor are used in the PAM stations.
# Plus the following approximation:
# Value to find in Kaimal and Finnigan 1994 p.224 (see comment in Andreas et al. 2010; Eq.3.4) corresponding to the approx of wthv by wths (from the sonic temp.)
wthv=0.05+0.05

# Wind speed:
# Typical from R3 Gill, to be ajusted
u=0.1
