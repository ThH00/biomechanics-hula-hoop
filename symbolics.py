import sympy as sp
import numpy as np

x3 = sp.symbols('x3')
tau = sp.symbols('tau')

# radius of hoop
R_hoop = sp.symbols('R_hoop')

E1 = np.array([1,0,0])
E2 = np.array([0,1,0])
E3 = np.array([0,0,1])

# Define an array of symbols
xbar_hoop = sp.symbols('xbar_hoop:3')   # Creates xbar_hoop0, xbar_hoop1, xbar_hoop2
xbar_hip = sp.symbols('xbar_hip:3')     # Creates xbar_hip0, xbar_hip1, xbar_hip2

# Defining velocity variables
vbar_hoop = sp.symbols('vbar_hoop:3')     # Creates vB0, vB1, vB2
vbar_hip = sp.symbols('vbar_hip:3')     # Creates vD0, vD1, vD2

# Defining acceleration variables
abar_hoop = sp.symbols('abar_hoop:3')     # Creates aB0, aB1, aB2
abar_hip = sp.symbols('abar_hip:3')     # Creates aD0, aD1, aD2

psi = sp.symbols('psi')
theta = sp.symbols('theta')
phi = sp.symbols('phi')

psidot = sp.symbols('psidot')
thetadot = sp.symbols('thetadot')
phidot = sp.symbols('phidot')

psiddot = sp.symbols('psiddot')
thetaddot = sp.symbols('thetaddot')
phiddot = sp.symbols('phiddot')

# Define the array of variables that are a function of time
# NOTE: I will just take derivative with respect vars, not tau and gamma
vars = np.array([xbar_hoop[0], xbar_hoop[1], xbar_hoop[2], psi, theta, phi, xbar_hip[0], xbar_hip[1], xbar_hip[2]])
varsdot = np.array([vbar_hoop[0], vbar_hoop[1], vbar_hoop[2], psidot, thetadot, phidot, vbar_hip[0], vbar_hip[1], vbar_hip[2]])
varsddot = np.array([abar_hoop[0], abar_hoop[1], abar_hoop[2], psiddot, thetaddot, phiddot, abar_hip[0], abar_hip[1], abar_hip[2]])

# Rotation matrices
R1 = np.array([[sp.cos(psi), sp.sin(psi), 0],[-sp.sin(psi), sp.cos(psi), 0],[0, 0, 1]])
R2 = np.array([[1, 0, 0],[0, sp.cos(theta), sp.sin(theta)],[0, -sp.sin(theta), sp.cos(theta)]])
R3 = np.array([[sp.cos(phi), sp.sin(phi), 0],[-sp.sin(phi), sp.cos(phi), 0],[0, 0, 1]])

e1p = np.transpose(R1)@E1
e1 = np.transpose(R3@R2@R1)@E1
e2 = np.transpose(R3@R2@R1)@E2
e3 = np.transpose(R3@R2@R1)@E3

omega_hip = sp.symbols('omega_hip:3')
omega_hoop = psidot*E3+thetadot*e1p+phidot*e3

## Slip speed constraint gradient
# right handed orthonormal basis of contact point
n = sp.cos(tau)*e1+sp.sin(tau)*e2
t1 = -sp.sin(tau)*e1+sp.cos(tau)*e2
t2 = np.cross(n,t1)

# position vector of point on hoop
xQ = xbar_hoop+R_hoop*n
    
# position vector of point on hip
# R_hip = 0.5-0.4*x3
R_hip = 0.2
xP = xbar_hip+R_hip*n

# Velocities of points P and Q
vQ = vbar_hoop+np.cross(omega_hoop,xQ-xbar_hoop)
vP = vbar_hip+np.cross(omega_hip,xP-xbar_hip)

# Gap distance constraints
gN = np.dot(xQ-xP,n)

# First derivative with respect to each variable
grad = [sp.diff(gN, xi) for xi in vars]
# Calculating contact constraint derivative
gNdot = 0
for i in range(9):
    gNdot += grad[i]*varsdot[i]

# Second derivative with respect to each variable
grad2 = [sp.diff(gN, xi, 2) for xi in vars]
# Calculating second constraint derivative
gNddot = 0
for i in range(9):
    gNddot += grad2[i]*(varsdot[i]**2)+grad[i]*varsddot[i]

# Calculating the constraint gradient
WN0 = grad[0]
WN1 = grad[1]
WN2 = grad[2] 
WN3 = grad[3]
WN4 = grad[4]
WN5 = grad[5]

# Slip speeds
gammaF1 = np.dot(vP-vQ,t1)
gammaF2 = np.dot(vP-vQ,t2)

WF1_0 = sp.diff(gammaF1, xbar_hoop[0])
WF1_1 = sp.diff(gammaF1, xbar_hoop[1])
WF1_2 = sp.diff(gammaF1, xbar_hoop[2])
WF1_3 = sp.diff(gammaF1, psi)
WF1_4 = sp.diff(gammaF1, theta)
WF1_5 = sp.diff(gammaF1, phi)

WF2_0 = sp.diff(gammaF2, xbar_hoop[0])
WF2_1 = sp.diff(gammaF2, xbar_hoop[1])
WF2_2 = sp.diff(gammaF2, xbar_hoop[2])
WF2_3 = sp.diff(gammaF2, psi)
WF2_4 = sp.diff(gammaF2, theta)
WF2_5 = sp.diff(gammaF2, phi)

## Preparing expressions for numpy

def prep_for_numpy(string):
    string = string.replace('sin', 'np.sin')
    string = string.replace('cos', 'np.cos')

    string = string.replace('xbar_hip0', 'xbar_hip[iter,0]')
    string = string.replace('xbar_hip1', 'xbar_hip[iter,1]')
    string = string.replace('xbar_hip2', 'xbar_hip[iter,2]')
    string = string.replace('xbar_hoop0', 'xbar_hoop[0]')
    string = string.replace('xbar_hoop1', 'xbar_hoop[1]')
    string = string.replace('xbar_hoop2', 'xbar_hoop[2]')

    string = string.replace('vbar_hip0', 'vbar_hip[iter,0]')
    string = string.replace('vbar_hip1', 'vbar_hip[iter,1]')
    string = string.replace('vbar_hip2', 'vbar_hip[iter,2]')
    string = string.replace('vbar_hoop0', 'vbar_hoop[0]')
    string = string.replace('vbar_hoop1', 'vbar_hoop[1]')
    string = string.replace('vbar_hoop2', 'vbar_hoop[2]')

    string = string.replace('abar_hip0', 'abar_hip[iter,0]')
    string = string.replace('abar_hip1', 'abar_hip[iter,1]')
    string = string.replace('abar_hip2', 'abar_hip[iter,2]')
    string = string.replace('abar_hoop0', 'abar_hoop[0]')
    string = string.replace('abar_hoop1', 'abar_hoop[1]')
    string = string.replace('abar_hoop2', 'abar_hoop[2]')

    string = string.replace('omega_hip0', 'omega_hip[0]')
    string = string.replace('omega_hip1', 'omega_hip[1]')
    string = string.replace('omega_hip2', 'omega_hip[2]')

    return(string)

gNdot = prep_for_numpy(str(gNdot))
gNddot = prep_for_numpy(str(gNddot))

WN0 = prep_for_numpy(str(WN0))
WN1 = prep_for_numpy(str(WN1))
WN2 = prep_for_numpy(str(WN2))
WN3 = prep_for_numpy(str(WN3))
WN4 = prep_for_numpy(str(WN4))
WN5 = prep_for_numpy(str(WN5))

WF1_0 = prep_for_numpy(str(WF1_0))
WF1_1 = prep_for_numpy(str(WF1_1))
WF1_2 = prep_for_numpy(str(WF1_2))
WF1_3 = prep_for_numpy(str(WF1_3))
WF1_4 = prep_for_numpy(str(WF1_4))
WF1_5 = prep_for_numpy(str(WF1_5))

WF2_0 = prep_for_numpy(str(WF2_0))
WF2_1 = prep_for_numpy(str(WF2_1))
WF2_2 = prep_for_numpy(str(WF2_2))
WF2_3 = prep_for_numpy(str(WF2_3))
WF2_4 = prep_for_numpy(str(WF2_4))
WF2_5 = prep_for_numpy(str(WF2_5))

## Writing expressions to file

# Define the string


# Open the file in append mode
with open("output.txt", "a") as file:
    # Append the string to the file
    file.write(f'gNdot = {gNdot}\n')
    file.write(f'gNddot = {gNddot}\n\n')

    file.write(f'WN[0,0] = {WN0}\n')
    file.write(f'WN[0,1] = {WN1}\n')
    file.write(f'WN[0,2] = {WN2}\n')
    file.write(f'WN[0,3] = {WN3}\n')
    file.write(f'WN[0,4] = {WN4}\n')
    file.write(f'WN[0,5] = {WN5}\n\n')

    file.write(f'WF[0,0] = {WF1_0}\n')
    file.write(f'WF[0,1] = {WF1_1}\n')
    file.write(f'WF[0,2] = {WF1_2}\n')
    file.write(f'WF[0,3] = {WF1_3}\n')
    file.write(f'WF[0,4] = {WF1_4}\n')
    file.write(f'WF[0,5] = {WF1_5}\n\n')

    file.write(f'WF[1,0] = {WF2_0}\n')
    file.write(f'WF[1,1] = {WF2_1}\n')
    file.write(f'WF[1,2] = {WF2_2}\n')
    file.write(f'WF[1,3] = {WF2_3}\n')
    file.write(f'WF[1,4] = {WF2_4}\n')
    file.write(f'WF[1,5] = {WF2_5}\n\n')
            

print('The End.')