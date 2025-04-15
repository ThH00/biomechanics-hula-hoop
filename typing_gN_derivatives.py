import numpy as np

# variables that need to be previously defined
# tau
# R_hoop
# R_hip
# E1, E2, E3
# xbar_hoop, vbar_hoop, abar_hoop
# xbar_hip, vbar_hip, abar_hip
# psi, theta, phi
# psidot, thetadot, phidot
# psiddot, thetaddot, phiddot
# R1, R2, R3
# e1p, e1, e2, e3


omega_hoop = psidot*E3+thetadot*e1p+phidot*e3

## Calculating the gap distance constraint, its derivative, and its gradient
u = sp.cos(tau)*e1+sp.sin(tau)*e2
xM = xbar_hoop+R_hoop*u
dv = np.dot(xM,E3)
temp = xM-dv*E3-xbar_hip
dh = np.dot(temp,temp)**0.5

gN = dh - R_hip

tempdot = xMdot-dvdot*E3-vbar_hip

xMdot = 

tempddot = 

gNdot = np.dot(temp,tempdot)/dh
gNddot = (np.dot(tempdot,tempdot)+np.dot(temp,tempddot)-gNdot**2)/dh