import sympy as sp
import numpy as np

tau = sp.symbols('tau') # parametrizing hoop

R_hoop = sp.symbols('R_hoop')   # radius of hoop

E1 = np.array([1,0,0])
E2 = np.array([0,1,0])
E3 = np.array([0,0,1])

## Define symbolic variables

# position vector to center base of hip
xbar_hip = sp.symbols('xbar_hip:3')     # Creates xbar_hip0, xbar_hip1, xbar_hip2
xbar_hoop = sp.symbols('xbar_hoop:3')   # Creates xbar_hoop0, xbar_hoop1, xbar_hoop2

vbar_hip = sp.symbols('vbar_hip:3')     # Creates vbar_hip0, vbar_hip1, vbar_hip2
vbar_hoop = sp.symbols('vbar_hoop:3')   # Creates vbar_hoop0, vbar_hoop1, vbar_hoop2

abar_hip = sp.symbols('abar_hip:3')     # Creates abar_hip0, abar_hip1, abar_hip2
abar_hoop = sp.symbols('abar_hoop:3')   # Creates abar_hoop0, abar_hoop1, abar_hoop2

psi_hip = sp.symbols('psi_hip')
theta_hip = sp.symbols('theta_hip')
phi_hip = sp.symbols('phi_hip')

psidot_hip = sp.symbols('psidot_hip')
thetadot_hip = sp.symbols('thetadot_hip')
phidot_hip = sp.symbols('phidot_hip')

psiddot_hip = sp.symbols('psiddot_hip')
thetaddot_hip= sp.symbols('thetaddot_hip')
phiddot_hip = sp.symbols('phiddot_hip')

psi_hoop = sp.symbols('psi_hoop')
theta_hoop = sp.symbols('theta_hoop')
phi_hoop = sp.symbols('phi_hoop')

psidot_hoop = sp.symbols('psidot_hoop')
thetadot_hoop = sp.symbols('thetadot_hoop')
phidot_hoop = sp.symbols('phidot_hoop')

psiddot_hoop = sp.symbols('psiddot_hoop')
thetaddot_hoop = sp.symbols('thetaddot_hoop')
phiddot_hoop = sp.symbols('phiddot_hoop')

# rotation matrices
R1_hip = np.array([[sp.cos(psi_hip), sp.sin(psi_hip), 0],[-sp.sin(psi_hip), sp.cos(psi_hip), 0],[0, 0, 1]])
R2_hip = np.array([[1, 0, 0],[0, sp.cos(theta_hip), sp.sin(theta_hip)],[0, -sp.sin(theta_hip), sp.cos(theta_hip)]])
R3_hip = np.array([[sp.cos(phi_hip), sp.sin(phi_hip), 0],[-sp.sin(phi_hip), sp.cos(phi_hip), 0],[0, 0, 1]])

R1_hoop = np.array([[sp.cos(psi_hoop), sp.sin(psi_hoop), 0],[-sp.sin(psi_hoop), sp.cos(psi_hoop), 0],[0, 0, 1]])
R2_hoop = np.array([[1, 0, 0],[0, sp.cos(theta_hoop), sp.sin(theta_hoop)],[0, -sp.sin(theta_hoop), sp.cos(theta_hoop)]])
R3_hoop = np.array([[sp.cos(phi_hoop), sp.sin(phi_hoop), 0],[-sp.sin(phi_hoop), sp.cos(phi_hoop), 0],[0, 0, 1]])

e1p_hip = np.transpose(R1_hip)@E1
e1_hip = np.transpose(R3_hip@R2_hip@R1_hip)@E1
e2_hip = np.transpose(R3_hip@R2_hip@R1_hip)@E2
e3_hip = np.transpose(R3_hip@R2_hip@R1_hip)@E3

e1p_hoop = np.transpose(R1_hoop)@E1
e1_hoop = np.transpose(R3_hoop@R2_hoop@R1_hoop)@E1
e2_hoop = np.transpose(R3_hoop@R2_hoop@R1_hoop)@E2
e3_hoop = np.transpose(R3_hoop@R2_hoop@R1_hoop)@E3

omega_hip = psidot_hip*E3+thetadot_hip*e1p_hip+phidot_hip*e3_hip
omega_hoop = psidot_hoop*E3+thetadot_hoop*e1p_hoop+phidot_hoop*e3_hoop

# Define the array of variables that are a function of time
# NOTE: I will just take derivative with respect vars, not tau
vars = np.array([xbar_hoop[0], xbar_hoop[1], xbar_hoop[2], psi_hoop, theta_hoop, phi_hoop, xbar_hip[0], xbar_hip[1], xbar_hip[2],psi_hip, theta_hip, phi_hip])
varsdot = np.array([vbar_hoop[0], vbar_hoop[1], vbar_hoop[2], psidot_hoop, thetadot_hoop, phidot_hoop, vbar_hip[0], vbar_hip[1], vbar_hip[2],psidot_hip, thetadot_hip, phidot_hip])
varsddot = np.array([abar_hoop[0], abar_hoop[1], abar_hoop[2], psiddot_hoop, thetaddot_hoop, phiddot_hoop, abar_hip[0], abar_hip[1], abar_hip[2],psiddot_hip, thetaddot_hip, phiddot_hip])
n_vars = np.size(vars)

## Calculate the gap distance constraint, its derivative, and its gradient for some tau
u = sp.cos(tau)*e1_hoop+sp.sin(tau)*e2_hoop
xM = xbar_hoop+R_hoop*u
dv = np.dot(xM-xbar_hip,e3_hip)
dh_vec = xM-dv*e3_hip-xbar_hip
dh = np.dot(dh_vec,dh_vec)**0.5
v = dh_vec/dh

def get_R_hip(dv):
    ''' In general, the radius is written as a function of dv
        # Example: hyperboloid
        # position vector of point on hip: R_hip = x1*e1_hip+y2*e2_hip+z3*e3_hip
        # equation of surface: S = x1^2/a^2+x2^2/a^2-x3^2/c^2-1
        a = 1
        c = 1
        R_hip = sp.sqrt(a**2+(a/c)**2*(dv**2))
        n = 2*x/a**2*Ex+2*y/a**2*Ey+2*z/c**2*Ez
    '''
    # Cylinderical hip
    R_hip = 0.2
    return R_hip

R_hip = get_R_hip(dv)
gN = dh - R_hip

# Get a right handed basis at contact point
xP_wrt_xbar_hip = dv*e3_hip+R_hip*v
xP_rel1 = np.dot(xP_wrt_xbar_hip,e1_hip)    # dh*dot(v,e1_hip)
xP_rel2 = np.dot(xP_wrt_xbar_hip,e2_hip)    # dh*dot(v,e2_hip)
xP_rel3 = np.dot(xP_wrt_xbar_hip,e3_hip)    # dv

def get_n(x1,x2,x3):
    '''Get unit normal to surface a specific point.
        equation of surface: S(x1,x2,x3)
        unit normal: n = S_x1*e1_hip+S_x2*e2_hip+S_x3*e3_hip
    '''
    n = np.array([x1, x2, 0])/((x1**2+x2**2)**0.5)
    return n

n = get_n(xP_rel1, xP_rel2, xP_rel3)
temp3 = np.cross(e3_hip,n)
t1 = temp3/(np.dot(temp3,temp3)**0.5)   # horizontal tangent
t2 = np.cross(n,t1)

# First derivative with respect to each variable
grad_N = [sp.diff(gN, xi) for xi in vars]
# Calculate contact constraint derivative
gNdot = 0
for i in range(n_vars):
    gNdot += grad_N[i]*varsdot[i]

# Second derivative with respect to each variable
grad2_N = [sp.diff(gN, xi, 2) for xi in vars]
# Calculate second constraint derivative
gNddot = 0
for i in range(n_vars):
    gNddot += grad2_N[i]*(varsdot[i]**2)+grad_N[i]*varsddot[i]

# Calculate the constraint gradient
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
xP = xbar_hip+dv*e3_hip+R_hip*v
# R_hip also a function of time
vP = vbar_hip+np.cross(omega_hip,dv*E3+R_hip*v)

# Slip speeds (applicable when M and P are touching)
gammaF1 = np.dot(vM-vP,t1)
gammaF2 = np.dot(vM-vP,t2)

# First derivative with respect to each variable
grad_F1 = [sp.diff(gammaF1, xi) for xi in varsdot]
grad_F2 = [sp.diff(gammaF2, xi) for xi in varsdot]
# Calculating contact constraints derivatives
gammadotF1 = 0
gammadotF2 = 0
for i in range(n_vars):
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

    string = string.replace('R_hoop', 'self.R_hoop')

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
# Open the file in append mode
with open("output_moving_cylinder.txt", "a") as file:
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