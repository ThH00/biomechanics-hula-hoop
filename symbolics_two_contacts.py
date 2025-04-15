import sympy as sp
import numpy as np

# assumptions
# hip has no angular velocity

# dh = sp.symbols('dh')       # horizontal distance from hoop base center to minimizing point
tau = sp.symbols('tau')

# radius of hoop
R_hoop = sp.symbols('R_hoop')

# radius of hip
R_hip = sp.symbols('R_hip')

E1 = np.array([1,0,0])
E2 = np.array([0,1,0])
E3 = np.array([0,0,1])

# Define an array of symbols
xbar_hoop = sp.symbols('xbar_hoop:3')   # Creates xbar_hoop0, xbar_hoop1, xbar_hoop2
xbar_hip = sp.symbols('xbar_hip:3')     # Creates xbar_hip0, xbar_hip1, xbar_hip2

# Defining velocity variables
vbar_hoop = sp.symbols('vbar_hoop:3')   # Creates vbar_hoop0, vbar_hoop1, vbar_hoop2
vbar_hip = sp.symbols('vbar_hip:3')     # Creates vbar_hip0, vbar_hip1, vbar_hip2

# Defining acceleration variables
abar_hoop = sp.symbols('abar_hoop:3')   # Creates abar_hoop0, abar_hoop1, abar_hoop2
abar_hip = sp.symbols('abar_hip:3')     # Creates abar_hip0, abar_hip1, abar_hip2

psi = sp.symbols('psi')
theta = sp.symbols('theta')
phi = sp.symbols('phi')

psidot = sp.symbols('psidot')
thetadot = sp.symbols('thetadot')
phidot = sp.symbols('phidot')

psiddot = sp.symbols('psiddot')
thetaddot = sp.symbols('thetaddot')
phiddot = sp.symbols('phiddot')

# Rotation matrices
R1 = np.array([[sp.cos(psi), sp.sin(psi), 0],[-sp.sin(psi), sp.cos(psi), 0],[0, 0, 1]])
R2 = np.array([[1, 0, 0],[0, sp.cos(theta), sp.sin(theta)],[0, -sp.sin(theta), sp.cos(theta)]])
R3 = np.array([[sp.cos(phi), sp.sin(phi), 0],[-sp.sin(phi), sp.cos(phi), 0],[0, 0, 1]])

e1p = np.transpose(R1)@E1
e1 = np.transpose(R3@R2@R1)@E1
e2 = np.transpose(R3@R2@R1)@E2
e3 = np.transpose(R3@R2@R1)@E3

omega_hoop = psidot*E3+thetadot*e1p+phidot*e3


omega_hip  = sp.symbols('omega_hip:3')  # Creates omega_hip0, omega_hip1, omega_hip2
alpha_hip = sp.symbols('alpha_hip:3')   # Creates alpha_hip0, alpha_hip1, alpha_hip2

hip_ang_vel = np.array([omega_hip[0],omega_hip[1],omega_hip[2]])
hip_ang_acc = np.array([alpha_hip[0],alpha_hip[1],alpha_hip[2]])

# Define the array of variables that are a function of time
# NOTE: I will just take derivative with respect vars, not tau
vars = np.array([xbar_hoop[0], xbar_hoop[1], xbar_hoop[2], psi, theta, phi, xbar_hip[0], xbar_hip[1], xbar_hip[2]])
varsdot = np.array([vbar_hoop[0], vbar_hoop[1], vbar_hoop[2], psidot, thetadot, phidot, vbar_hip[0], vbar_hip[1], vbar_hip[2]],omega_hip[0],omega_hip[1],omega_hip[2])
varsddot = np.array([abar_hoop[0], abar_hoop[1], abar_hoop[2], psiddot, thetaddot, phiddot, abar_hip[0], abar_hip[1], abar_hip[2]],alpha_hip[0],alpha_hip[1],alpha_hip[2])
n_vars = np.size(vars)

## Calculating the gap distance constraint, its derivative, and its gradient
u = sp.cos(tau)*e1+sp.sin(tau)*e2
xM = xbar_hoop+R_hoop*u
dv = np.dot(xM,E3)
temp = xM-dv*E3-xbar_hip
dh = np.dot(temp,temp)**0.5

gN = dh - R_hip

# Right handed orthonormal basis of contact point
v = temp/dh
t1 = E3
t2 = np.cross(v,t1)

# First derivative with respect to each variable
grad_N = [sp.diff(gN, xi) for xi in vars]
# Calculating contact constraint derivative
gNdot = 0
for i in range(n_vars):
    gNdot += grad_N[i]*varsdot[i]

# Second derivative with respect to each variable
grad2_N = [sp.diff(gN, xi, 2) for xi in vars]
# Calculating second constraint derivative
gNddot = 0
for i in range(n_vars):
    gNddot += grad2_N[i]*(varsdot[i]**2)+grad_N[i]*varsddot[i]

# Calculating the constraint gradient
WN0 = grad_N[0]
WN1 = grad_N[1]
WN2 = grad_N[2] 
WN3 = grad_N[3]
WN4 = grad_N[4]
WN5 = grad_N[5]

## Calculating the slip speed constraints, their derivatives, and their constraint gradients

# Motion of contact point on hoop
xM = xbar_hoop+R_hoop*u
vM = vbar_hoop+np.cross(omega_hoop,R_hoop*u)

# Motion of contact point on hip
xP = xbar_hip+dv*E3+R_hip*v
vP = vbar_hip+np.cross(hip_ang_vel,dv*E3+R_hip*v)

# Slip speeds
gammaF1 = np.dot(vM-vP,t1)
gammaF2 = np.dot(vM-vP,t2)

# First derivative with respect to each variable
grad_F1 = [sp.diff(gammaF1, xi) for xi in varsdot]
grad_F2 = [sp.diff(gammaF2, xi) for xi in varsdot]
# Calculating contact constraint derivative
gammadotF1 = 0
gammadotF2 = 0
for i in range(n_vars+3):
    gammadotF1 += grad_F1[i]*varsddot[i]
    gammadotF2 += grad_F2[i]*varsddot[i]

WF1_0 = grad_F1[0]
WF1_1 = grad_F1[1]
WF1_2 = grad_F1[2]
WF1_3 = grad_F1[3]
WF1_4 = grad_F1[4]
WF1_5 = grad_F1[5]

WF2_0 = grad_F2[0]
WF2_1 = grad_F2[1]
WF2_2 = grad_F2[2]
WF2_3 = grad_F2[3]
WF2_4 = grad_F2[4]
WF2_5 = grad_F2[5]

## Preparing expressions for numpy

def prep_for_numpy(string):
    string = string.replace('sin', 'np.sin')
    string = string.replace('cos', 'np.cos')

    string = string.replace('xbar_hip0', 'xbar_hip[0]')
    string = string.replace('xbar_hip1', 'xbar_hip[1]')
    string = string.replace('xbar_hip2', 'xbar_hip[2]')
    string = string.replace('xbar_hoop0', 'xbar_hoop[0]')
    string = string.replace('xbar_hoop1', 'xbar_hoop[1]')
    string = string.replace('xbar_hoop2', 'xbar_hoop[2]')

    string = string.replace('vbar_hip0', 'vbar_hip[0]')
    string = string.replace('vbar_hip1', 'vbar_hip[1]')
    string = string.replace('vbar_hip2', 'vbar_hip[2]')
    string = string.replace('vbar_hoop0', 'vbar_hoop[0]')
    string = string.replace('vbar_hoop1', 'vbar_hoop[1]')
    string = string.replace('vbar_hoop2', 'vbar_hoop[2]')

    string = string.replace('abar_hip0', 'abar_hip[0]')
    string = string.replace('abar_hip1', 'abar_hip[1]')
    string = string.replace('abar_hip2', 'abar_hip[2]')
    string = string.replace('abar_hoop0', 'abar_hoop[0]')
    string = string.replace('abar_hoop1', 'abar_hoop[1]')
    string = string.replace('abar_hoop2', 'abar_hoop[2]')

    string = string.replace('omega_hip0', 'omega_hip[0]')
    string = string.replace('omega_hip1', 'omega_hip[1]')
    string = string.replace('omega_hip2', 'omega_hip[2]')

    string = string.replace('alpha_hip0', 'alpha_hip[0]')
    string = string.replace('alpha_hip1', 'alpha_hip[1]')
    string = string.replace('alpha_hip2', 'alpha_hip[2]')

    return(string)

gN = prep_for_numpy(str(gN))
gNdot = prep_for_numpy(str(gNdot))
gNddot = prep_for_numpy(str(gNddot))

WN0 = prep_for_numpy(str(WN0))
WN1 = prep_for_numpy(str(WN1))
WN2 = prep_for_numpy(str(WN2))
WN3 = prep_for_numpy(str(WN3))
WN4 = prep_for_numpy(str(WN4))
WN5 = prep_for_numpy(str(WN5))

gammaF1 = prep_for_numpy(str(gammaF1))
gammaF2 = prep_for_numpy(str(gammaF2))
gammadotF1 = prep_for_numpy(str(gammadotF1))
gammadotF2 = prep_for_numpy(str(gammadotF2))

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
    file.write(f'gN = {gN}\n')
    file.write(f'gNdot = {gNdot}\n')
    file.write(f'gNddot = {gNddot}\n\n')

    file.write(f'WN[0,0] = {WN0}\n')
    file.write(f'WN[0,1] = {WN1}\n')
    file.write(f'WN[0,2] = {WN2}\n')
    file.write(f'WN[0,3] = {WN3}\n')
    file.write(f'WN[0,4] = {WN4}\n')
    file.write(f'WN[0,5] = {WN5}\n\n')

    file.write(f'gammaF1 = {gammaF1}\n')
    file.write(f'gammaF2 = {gammaF2}\n')
    file.write(f'gammadotF1 = {gammadotF1}\n')
    file.write(f'gammadotF2 = {gammadotF2}\n\n')

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