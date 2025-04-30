# simulating the motion of a hula hoop

## Importing packages
import numpy as np
import time
import os
from scipy.signal import argrelextrema
import scipy.io

start_time = time.time()

output_path = os.path.join(os.getcwd(), "outputs/multiple_solutions")
os.makedirs(output_path, exist_ok=True)

# Specify the maximum duration in hours for one run 
max_hours = 50

# Specify the maximum number of leaves beyond which the code will stop running (actuall max leaves is max_leaves+1)
max_leaves = 2


# creating custom exceptions
class MaxNewtonIterAttainedError(Exception):
    """This exception is raised when the maximum number of Newton iterations is attained
      whilst the iterations have not yet converged and the solution was not yet obtained."""
    def __init__(self, message="This exception is raised when the maximum number of Newton iterations is attained."):
        self.message = message
        super().__init__(self.message)

class RhoInfInfiniteLoop(Exception):
    """This exception is raised when we have possibly entered in an infinite loop through updating rho_inf."""
    def __init__(self, message="This exception is raised when we have possibly entered in an infinite loop through updating rho_inf."):
        self.message = message
        super().__init__(self.message)

class MaxHoursAttained(Exception):
    """This exception is raised when the maximum number of run hours specified by the use is exceeded."""
    def __init__(self, message="This exception is raised when the maximum run time is exceeded."):
        self.message = message
        super().__init__(self.message)

class MaxLeavesAttained(Exception):
    """This exception is raised when the maximum number of run leaves specified by the use is exceeded."""
    def __init__(self, message="This exception is raised when the maximum number of leaves is exceeded."):
        self.message = message
        super().__init__(self.message)

class NoLocalMinima(Exception):
    def __init__(self, message="The distance between the hoop and the hip has less then 1 or more than 2 local minima."):
        super().__init__(message)

# f is an output file that logs failed runs
# it is saved in the directory containing the output file
directory_containing_output = os.path.dirname(output_path)
f = open(f"{directory_containing_output}/run_failures_log.txt",'a')

# g details the birth and death of multiple solutions
g = open(f"{output_path}/bifurcation_map.txt",'w') 



## Problem constants
gr = 9.81           # m/s^2, gravitational acceleration

## Simulation parameters
ntime = 2000        # dimensionless, number of iterations
dtime = 2e-3        # s, time step duration
t_arr = np.arange(0, ntime*dtime, dtime)

## Hip axis properties
R_hip = 0.2

# The hip center is tracing an ellipse
# Position of the bottom center of hip (bottom of hip axis)
x1bar_hip = 1*np.ones(ntime)*(t_arr**2)/2
x2bar_hip = np.zeros(ntime)+0.1
xbar_hip = np.column_stack((x1bar_hip, x2bar_hip, np.zeros(ntime)))
# velocity of the bottom center of hip
v1bar_hip = 1*np.ones(ntime)*t_arr
v2bar_hip = np.zeros(ntime)
vbar_hip = np.column_stack((v1bar_hip, v2bar_hip, np.zeros(ntime)))
# acceleration of the bottom center of hip
a1bar_hip = 1*np.ones(ntime)
a2bar_hip = np.zeros(ntime)
abar_hip = np.column_stack((a1bar_hip, a2bar_hip, np.zeros(ntime)))

# # The hip center is tracing an ellipse
# # Position of the bottom center of hip (bottom of hip axis)
# x1bar_hip = 0.2*np.cos(5*t_arr)
# x2bar_hip = 0.6*np.sin(5*t_arr)
# xbar_hip = np.column_stack((x1bar_hip, x2bar_hip, np.zeros(ntime)))
# # velocity of the bottom center of hip
# v1bar_hip = -0.2*5*np.sin(5*t_arr)
# v2bar_hip = 0.6*5*np.cos(5*t_arr)
# vbar_hip = np.column_stack((v1bar_hip, v2bar_hip, np.zeros(ntime)))
# # acceleration of the bottom center of hip
# a1bar_hip = -0.2*25*np.cos(5*t_arr)
# a2bar_hip = -0.6*25*np.sin(5*t_arr)
# abar_hip = np.column_stack((a1bar_hip, a2bar_hip, np.zeros(ntime)))

# # The hip center is fixed
# # Position of the bottom center of hip (bottom of hip aixs)
# xbar_hip = np.zeros((ntime,3))
# # velocity of the bottom center of hip
# vbar_hip = np.zeros((ntime,3))
# # acceleration of the bottom center of hip
# abar_hip = np.zeros((ntime,3))

# Angular velocity and angular acceleration of hip
# omega_hip = np.array([0,0,1])   # angular velocity of hip
omega_hip = np.array([0,0,1])   # angular velocity of hip
alpha_hip = np.array([0,0,0])   # angular acceleration of hip

g.write(f"######\n Ruunning a simulation with the hip tracing a straight line:\n")
g.write(f"      abar_hip = {a1bar_hip[0]} Ex+ {a2bar_hip[0]} Ey.\n")
g.write(f"    The hip is rotating with an angular velocity omega_hip = {omega_hip}.\n")
g.write(f"    Total duration of simulation: {dtime*ntime}.\n")

# hoop properties
ndof = 6                # number of degrees of freedom of the hoop
R_hoop = 0.5            # m, radius of hoop
m = 0.2                 # kg, mass of hoop
It = 0.5*m*R_hoop**2    # kg.m^2, rotational inertia of hoop about diameter
Ia = m*R_hoop**2        # kg.m^2, rotational inertia of hoop about axis passing through center perp to hoop plane

# restitution coefficients
eN = 0                # dimensionless, normal impact restitution coefficient
eF = 0                  # dimensionless, tangential impact restitution coefficient
g.write(f"    eN = {eN}.\n")

# friction coefficients
mu_s = 0.4              # dimensionless, static friction coefficient
mu_k = 0.2              # dimensionless, kinetic friction coefficient
g.write(f"    mu_s = {mu_s}, mu_k = {mu_k}.\n\n")

# constraint count
ng = 0                  # number of position level constraints
nN = 2                  # number of no penetration contact constraints
ngamma = 0              # number of velocity level constraints
nF = 4                  # slip speed constraints/friction force
gammaF_lim = np.array([[0,1],[2,3]])

# fixed basis vectors
E1 = np.array([1,0,0])
E2 = np.array([0,1,0])
E3 = np.array([0,0,1])

# generalized alpha parameters
r = 0.3     # approximation parameter
rho_inf = 0.5
rho_infinity_initial = rho_inf
# eq. 72
alpha_m = (2*rho_inf-1)/(rho_inf+1)
alpha_f = rho_inf/(rho_inf+1)
gama = 0.5+alpha_f-alpha_m
beta = 0.25*(0.5+gama)**2

# loop parameters
maxiter_n = 20
tol_n = 1.0e-6

# mass matrix (constant)
Mdiag = np.array([m, m, m, It, It, Ia])
M = np.diag(Mdiag)

# vector of applied forces and moments (weight)
# f = np.array([0, 0, -m*gr, 0, 0, 0])      # I removed gravity for now
f = np.array([0, 0, 0, 0, 0, 0])

# initialize arrays to save results
R_array = np.zeros(ntime)

gammaF_save = np.zeros((ntime,nF))
gNdot_save = np.zeros((ntime,nN))

q = np.zeros((ntime,ndof))
u = np.zeros((ntime,ndof))

# initial conditions
# q0 = np.array([0.05, 0.1, 1., 0., np.pi.6, 0.])
# u0 = np.array([1, 0, 0, 0, 3, 0])
q0 = np.array([0, 0, 1, 0, 0, 0])
u0 = np.array([0, 0, 0, 0, 0, 1])

# number of algorithm degrees of freedom
nX = 3*ndof+3*ng+2*ngamma+3*nN+2*nF
x0 = np.zeros(nX)

# initial auxiliary variables
a_bar0 = np.zeros(ndof)
lambdaN_bar0 = np.zeros(nN)
lambdaF_bar0 = np.zeros(nF)

prev_AV = np.concatenate((a_bar0, lambdaN_bar0, lambdaF_bar0), axis=None)

gammaF0 = np.zeros(nF)
gNdot0 = np.zeros(nN)

gN_save = np.zeros((nN, ntime))
x_save = np.zeros((nX, ntime))
minimizing_tau_save = np.zeros((2,ntime))

def update_rho_inf():
    global rho_inf, alpha_m, alpha_f, gama, beta, iter
    rho_inf = rho_inf+0.05  #0.01
    print(rho_inf)
    if np.abs(rho_inf - rho_infinity_initial) < 0.001:
        g.write("\n     rho_inf has been updated through a complete cycle.\n")
        print(f"Iteration {iter}: possibility of infinite loop.\n")
        raise RhoInfInfiniteLoop
    if rho_inf > 1.001:
        rho_inf = 0
    # eq. 72
    alpha_m = (2*rho_inf-1)/(rho_inf+1)
    alpha_f = rho_inf/(rho_inf+1)
    gama = 0.5+alpha_f-alpha_m
    beta = 0.25*(0.5+gama)**2


def save_arrays():
    
    # file_name_J = str(f'{output_path}/J.mat')
    # scipy.io.savemat(file_name_J,dict(J=J))

    file_name_q = str(f'{output_path}/q.mat')
    file_name_u = str(f'{output_path}/u.mat')
    scipy.io.savemat(file_name_q,dict(q=q))
    scipy.io.savemat(file_name_u,dict(u=u))

    file_name_x_save = str(f'{output_path}/x_save.mat')
    scipy.io.savemat(file_name_x_save,dict(x=x_save))

    file_name_gN = str(f'{output_path}/gN.mat')
    scipy.io.savemat(file_name_gN,dict(gN=gN_save))

    file_name_xbar_hip = str(f'{output_path}/xbar_hip.mat')
    scipy.io.savemat(file_name_xbar_hip,dict(xbar_hip=xbar_hip))

    file_name_tau = str(f'{output_path}/tau.mat')
    scipy.io.savemat(file_name_tau,dict(tau=minimizing_tau_save))

    # np.save(f'{output_path}/q_save.npy', q_save)
    # np.save(f'{output_path}/u_save.npy', u_save)
    # np.save(f'{output_path}/X_save.npy', X_save)
    # np.save(f'{output_path}/gNdot_save.npy', gNdot_save)
    # np.save(f'{output_path}/gammaF_save.npy', gammaF_save)
    # np.save(f'{output_path}/AV_save.npy', AV_save)

    return

def get_x_components(x):
    a = x[0:ndof]
    U = x[ndof:2*ndof]
    Q = x[2*ndof:3*ndof]
    Kappa_g = x[3*ndof:3*ndof+ng]
    Lambda_g = x[3*ndof+ng:3*ndof+2*ng]
    lambda_g = x[3*ndof+2*ng:3*ndof+3*ng]
    Lambda_gamma = x[3*ndof+3*ng:3*ndof+3*ng+ngamma]
    lambda_gamma = x[3*ndof+3*ng+ngamma:3*ndof+3*ng+2*ngamma]
    Kappa_N = x[3*ndof+3*ng+2*ngamma:3*ndof+3*ng+2*ngamma+nN]
    Lambda_N = x[3*ndof+3*ng+2*ngamma+nN:3*ndof+3*ng+2*ngamma+2*nN]
    lambda_N = x[3*ndof+3*ng+2*ngamma+2*nN:3*ndof+3*ng+2*ngamma+3*nN]
    Lambda_F = x[3*ndof+3*ng+2*ngamma+3*nN:3*ndof+3*ng+2*ngamma+3*nN+nF]
    lambda_F = x[3*ndof+3*ng+2*ngamma+3*nN+nF:3*ndof+3*ng+2*ngamma+3*nN+2*nF]
    return a, U, Q, Kappa_g, Lambda_g, lambda_g, Lambda_gamma, lambda_gamma, Kappa_N, Lambda_N, lambda_N, Lambda_F, lambda_F

def sign_no_zero(x):
    return np.where(x >= 0, 1, -1)

def get_minimizing_tau(q, xbar_hip):

    # center of hoop
    xbar_hoop = q[:3]
    # Euler angles of hoop
    psi = q[3]
    theta = q[4]
    phi = q[5]
    # Rotation matrices
    R1 = np.array([[np.cos(psi), np.sin(psi), 0],[-np.sin(psi), np.cos(psi), 0],[0, 0, 1]])
    R2 = np.array([[1, 0, 0],[0, np.cos(theta), np.sin(theta)],[0, -np.sin(theta), np.cos(theta)]])
    R3 = np.array([[np.cos(phi), np.sin(phi), 0],[-np.sin(phi), np.cos(phi), 0],[0, 0, 1]])
    # {E1, E2, E3} components
    e1 = np.transpose(R3@R2@R1)@E1
    e2 = np.transpose(R3@R2@R1)@E2

    # Create an array of possible tau values (step size < algorithm tolerance)
    tau = np.linspace(0, 2*np.pi, num=n_tau, endpoint=True)
    # I can find intervals containing the minima and then refine the discretization in these intervals (or use the bisection method)

    # Creating array of hoop points
    # # Reshape tau to (1000000, 1) to enable broadcasting
    u = np.cos(tau)[:, np.newaxis] * e1 + np.sin(tau)[:, np.newaxis] * e2  # Shape (1000000, 3)

    xM = xbar_hoop+R_hoop*u

    # Calculating the value of dH for each point
    dv = np.dot(xM,E3)
    temp = xM-dv[:, np.newaxis]*E3-xbar_hip
    # Compute the norm of each row
    dh = np.linalg.norm(temp, axis=1)

    # dv = np.zeros((n_tau,1))
    # dh = np.zeros((n_tau,1))
    # for i in range(n_tau):
    #     dv[i] = np.dot(xM[i,:],E3)
    #     temp = xM[i,:]-dv[i]*E3-xbar_hip
    #     # Compute the norm of each row
    #     dh[i] = np.linalg.norm(temp)

    # Find the minimizers of dh
    # Find local minima (less than neighbors)
    min_indices = argrelextrema(dh, np.less)[0]
    # Find the minizing value of tau
    minimizing_tau = tau[min_indices]

    return minimizing_tau

def get_contact_constraints(q,u,a,tau,xbar_hip,vbar_hip,abar_hip):
    # gets gap distance, slip speed functions and their gradients and derivatives at each contact

    # prev_gNdot = gNdot_save[iter-1,:]

    # center of hoop
    xbar_hoop = q[:3]
    vbar_hoop = u[:3]
    abar_hoop = a[:3]
    # Euler angles of hoop
    psi = q[3]
    theta = q[4]
    phi = q[5]
    psidot = u[3]
    thetadot = u[4]
    phidot = u[5]
    psiddot = a[3]
    thetaddot = a[4]
    phiddot = a[5]
    
    WN = np.zeros((1,ndof))
    WF = np.zeros((2, ndof))

    # maybe make these rotation matrices and vectors global

    # Rotation matrices
    R1 = np.array([[np.cos(psi), np.sin(psi), 0],[-np.sin(psi), np.cos(psi), 0],[0, 0, 1]])
    R2 = np.array([[1, 0, 0],[0, np.cos(theta), np.sin(theta)],[0, -np.sin(theta), np.cos(theta)]])
    R3 = np.array([[np.cos(phi), np.sin(phi), 0],[-np.sin(phi), np.cos(phi), 0],[0, 0, 1]])
    # {E1, E2, E3} components
    e1p = np.transpose(R1)@E1
    e1 = np.transpose(R3@R2@R1)@E1
    e2 = np.transpose(R3@R2@R1)@E2
    e3 = np.transpose(R3@R2@R1)@E3

    omega_hoop = psidot*E3+thetadot*e1p+phidot*e3

    e1pdot = np.cross(psidot*E3,e1p)
    e3dot = np.cross(omega_hoop,e3)

    alpha_hoop = psiddot*E3+thetaddot*e1p+phiddot*e3+thetadot*e1pdot+phidot*e3dot

    tau_dot = 0
    tau_ddot = 0

    u_corrotational = tau_dot*(-np.sin(tau)*e1+np.cos(tau)*e2)
    u_double_corrotational = tau_ddot*(-np.sin(tau)*e1+np.cos(tau)*e2)

    u = np.cos(tau)*e1+np.sin(tau)*e2
    udot = u_corrotational+np.cross(omega_hoop,u)
    uddot = u_double_corrotational+2*np.cross(omega_hoop,u_corrotational)+np.cross(omega_hoop,np.cross(omega_hoop,u))+np.cross(alpha_hoop,u)

    xM = xbar_hoop+R_hoop*u
    vM = vbar_hoop+R_hoop*udot
    aM = abar_hoop+R_hoop*uddot

    # vertical components of vector from hip center to point on hoop
    H = xM-xbar_hip-np.dot(xM-xbar_hip,E3)*E3
    H_dot = vM-vbar_hip-np.dot(vM-vbar_hip,E3)*E3
    H_ddot = aM-abar_hip-np.dot(aM-abar_hip,E3)*E3

    norm_H = np.linalg.norm(H)
    gN = norm_H-R_hip
    gNdot = np.dot(H,H_dot)/norm_H
    gNddot = (np.dot(H_dot,H_dot)+np.dot(H,H_ddot))/norm_H

    WN[0,0] = 2*np.dot(H,E1)
    WN[0,1] = 2*np.dot(H,E2)
    WN[0,2] = 0

    de1_dpsi = np.cos(phi)*(E2*np.cos(psi) - E1*np.sin(psi)) - np.cos(theta)*np.sin(phi)*(E1*np.cos(psi) + E2*np.sin(psi))
    de2_dpsi = - np.sin(phi)*(E2*np.cos(psi) - E1*np.sin(psi)) - np.cos(phi)*np.cos(theta)*(E1*np.cos(psi) + E2*np.sin(psi))
    de1_dtheta = np.sin(phi)*(E3*np.cos(theta) - np.sin(theta)*(E2*np.cos(psi) - E1*np.sin(psi)))
    de2_dtheta = np.cos(phi)*(E3*np.cos(theta) - np.sin(theta)*(E2*np.cos(psi) - E1*np.sin(psi)))
    de1_dphi = np.cos(phi)*(E3*np.sin(theta) + np.cos(theta)*(E2*np.cos(psi) - E1*np.sin(psi))) - np.sin(phi)*(E1*np.cos(psi) + E2*np.sin(psi))
    de2_dphi = - np.cos(phi)*(E1*np.cos(psi) + E2*np.sin(psi)) - np.sin(phi)*(E3*np.sin(theta) + np.cos(theta)*(E2*np.cos(psi) - E1*np.sin(psi)))

    dxM_dpsi = np.cos(tau)*de1_dpsi+np.sin(tau)*de2_dpsi
    dxM_dtheta = np.cos(tau)*de1_dtheta+np.sin(tau)*de2_dtheta
    dxM_dphi = np.cos(tau)*de1_dphi+np.sin(tau)*de2_dphi

    WN[0,3] = 2*np.dot(H,dxM_dpsi-np.dot(dxM_dpsi,E3)*E3)
    WN[0,4] = 2*np.dot(H,dxM_dtheta-np.dot(dxM_dtheta,E3)*E3)
    WN[0,5] = 2*np.dot(H,dxM_dphi-np.dot(dxM_dphi,E3)*E3)

    # # comparing with previous values
    # different constraint formulation: there is a scaling difference in WN
    # WN0 = np.zeros((1,ndof))

    # gN0 = -R_hip + (xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    # gNdot0   = phidot*(1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + psidot*(1.0*R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + thetadot*(1.0*R_hoop*(np.sin(phi)*np.sin(psi)*np.sin(theta)*np.cos(tau) + np.sin(psi)*np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*(-np.sin(phi)*np.sin(theta)*np.cos(psi)*np.cos(tau) - np.sin(tau)*np.sin(theta)*np.cos(phi)*np.cos(psi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + vbar_hip[0]*(-1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) + 1.0*xbar_hip[0] - 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + vbar_hip[1]*(-1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + 1.0*xbar_hip[1] - 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + 1.0*vbar_hip[2]*xbar_hip[2]/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + vbar_hoop[0]*(1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - 1.0*xbar_hip[0] + 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + vbar_hoop[1]*(1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - 1.0*xbar_hip[1] + 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    # gNddot0 = 1.0*R_hoop*phidot**2*(-R_hoop*(((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.sin(tau))*(-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]) + ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) - (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))*(-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]))**2/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.sin(tau))**2 + R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) - (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))**2 - ((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1]) - ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**0.5) + 1.0*R_hoop*psidot**2*(-R_hoop*(((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) - ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))*(-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))*(-((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0]) + ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))**2 + R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))**2 - ((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1]) - ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**0.5) + 1.0*R_hoop*thetadot**2*(np.sin(phi)*np.cos(tau) + np.sin(tau)*np.cos(phi))*(-R_hoop*(-(-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*np.cos(psi) + (-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*np.sin(psi))*((R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])*np.cos(psi) - (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])*np.sin(psi))*(np.sin(phi)*np.cos(tau) + np.sin(tau)*np.cos(phi))*np.sin(theta)**2/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (R_hoop*(np.sin(phi)*np.cos(tau) + np.sin(tau)*np.cos(phi))*np.sin(psi)**2*np.sin(theta)**2 + R_hoop*(np.sin(phi)*np.cos(tau) + np.sin(tau)*np.cos(phi))*np.sin(theta)**2*np.cos(psi)**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])*np.cos(psi)*np.cos(theta) - (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])*np.sin(psi)*np.cos(theta))/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**0.5) + abar_hip[0]*(-1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) + 1.0*xbar_hip[0] - 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hip[1]*(-1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + 1.0*xbar_hip[1] - 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + 1.0*abar_hip[2]*xbar_hip[2]/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hoop[0]*(1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - 1.0*xbar_hip[0] + 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hoop[1]*(1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - 1.0*xbar_hip[1] + 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + phiddot*(1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + psiddot*(1.0*R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + thetaddot*(1.0*R_hoop*(np.sin(phi)*np.sin(psi)*np.sin(theta)*np.cos(tau) + np.sin(psi)*np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*(-np.sin(phi)*np.sin(theta)*np.cos(psi)*np.cos(tau) - np.sin(tau)*np.sin(theta)*np.cos(phi)*np.cos(psi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + 1.0*vbar_hip[0]**2*((-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5)) + 1.0*vbar_hip[1]**2*((-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5)) + 1.0*vbar_hip[2]**2*(-xbar_hip[2]**2/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5)) + 1.0*vbar_hoop[0]**2*((-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5)) + 1.0*vbar_hoop[1]**2*((-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5))

    # WN0[0,0] = (1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - 1.0*xbar_hip[0] + 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    # WN0[0,1] = (1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - 1.0*xbar_hip[1] + 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    # WN0[0,2] = 0
    # WN0[0,3] = (1.0*R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    # WN0[0,4] = (1.0*R_hoop*(np.sin(phi)*np.sin(psi)*np.sin(theta)*np.cos(tau) + np.sin(psi)*np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*(-np.sin(phi)*np.sin(theta)*np.cos(psi)*np.cos(tau) - np.sin(tau)*np.sin(theta)*np.cos(phi)*np.cos(psi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    # WN0[0,5] = (1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5

    gammaF1 = -R_hip*omega_hip[0]*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hip*omega_hip[1]*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(phidot*np.sin(psi)*np.sin(theta) + thetadot*np.cos(psi)) - R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(-phidot*np.sin(theta)*np.cos(psi) + thetadot*np.sin(psi)) - vbar_hip[2] + vbar_hoop[2]
    gammaF2 = (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*(R_hip*omega_hip[2]*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - R_hoop*(phidot*np.cos(theta) + psidot)*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + R_hoop*(-phidot*np.sin(theta)*np.cos(psi) + thetadot*np.sin(psi))*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) - omega_hip[1]*(-R_hip*xbar_hip[2]/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) + xbar_hoop[2]) - vbar_hip[0] + vbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*(-R_hip*omega_hip[2]*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hoop*(phidot*np.cos(theta) + psidot)*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - R_hoop*(phidot*np.sin(psi)*np.sin(theta) + thetadot*np.cos(psi))*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) + omega_hip[0]*(-R_hip*xbar_hip[2]/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) + xbar_hoop[2]) - vbar_hip[1] + vbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    gammadotF1 = -R_hip*alpha_hip[0]*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hip*alpha_hip[1]*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - abar_hip[2] + abar_hoop[2] + phiddot*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.sin(psi)*np.sin(theta) + R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(theta)*np.cos(psi)) + thetaddot*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(psi) - R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(psi))
    gammadotF2 = -abar_hip[0]*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hip[1]*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hoop[0]*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - abar_hoop[1]*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - alpha_hip[0]*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*(-R_hip*xbar_hip[2]/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) + xbar_hoop[2])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + alpha_hip[1]*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*(R_hip*xbar_hip[2]/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) - xbar_hoop[2])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + alpha_hip[2]*(R_hip*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**1.0 + R_hip*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**1.0) + phiddot*((-R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(theta) - R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(theta)*np.cos(psi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.cos(theta) - R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(psi)*np.sin(theta))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5) + psiddot*(-R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5) + thetaddot*(R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*np.sin(psi)/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*np.cos(psi)/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5)

    WF[0,0] = 0
    WF[0,1] = 0
    WF[0,2] = 1
    WF[0,3] = 0
    WF[0,4] = R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(psi) - R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(psi)
    WF[0,5] = R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.sin(psi)*np.sin(theta) + R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(theta)*np.cos(psi)

    WF[1,0] = (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    WF[1,1] = -(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    WF[1,2] = 0
    WF[1,3] = -R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    WF[1,4] = R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*np.sin(psi)/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*np.cos(psi)/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    WF[1,5] = (-R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(theta) - R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(theta)*np.cos(psi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.cos(theta) - R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(psi)*np.sin(theta))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5

    gammaF = np.array([gammaF1, gammaF2])
    gammadotF = np.array([gammadotF1, gammadotF2])
    
    return gN, gNdot, gNddot, WN, gammaF, gammadotF, WF

def combine_contact_constraints(q,u,a):
    # combine all gap distance, slip speed functions and the gradients and derivatives from both contacts

    # get the minimizing values
    tau = get_minimizing_tau(q,xbar_hip[iter,:])
    
    # Contact constraints and constraint gradients
    # initializing constraints
    gN = np.zeros(nN)
    gammaF = np.zeros(nF)
    # initialize constraint derivatives
    gNdot = np.zeros(nN)
    gNddot = np.zeros(nN)
    gammadotF = np.zeros(nF)
    # initializing constraint gradients
    WN = np.zeros((nN, ndof))
    WF = np.zeros((nF, ndof))

    if np.size(tau) == 2:  # two local minima
        gN[0], gNdot[0], gNddot[0], WN[0,:], gammaF[gammaF_lim[0,:]], gammadotF[gammaF_lim[0,:]], WF[gammaF_lim[0,:],:] = get_contact_constraints(q,u,a,tau[0],xbar_hip[iter,:],vbar_hip[iter,:],abar_hip[iter,:])
        gN[1], gNdot[1], gNddot[1], WN[1,:], gammaF[gammaF_lim[1,:]], gammadotF[gammaF_lim[1,:]], WF[gammaF_lim[1,:],:] = get_contact_constraints(q,u,a,tau[1],xbar_hip[iter,:],vbar_hip[iter,:],abar_hip[iter,:])
        # saving values
        minimizing_tau_save[:,iter] = tau 
        
    elif np.size(tau) == 1:
        # This case is rare if the hoop is not initialized to a horizontal configuration
        gN[0], gNdot[0], gNddot[0], WN[0,:], gammaF[gammaF_lim[0,:]], gammadotF[gammaF_lim[0,:]], WF[gammaF_lim[0,:],:] = get_contact_constraints(q,u,a,tau[0],xbar_hip[iter,:],vbar_hip[iter,:],abar_hip[iter,:])
        gN[1] = 1   # >0, no contact, we don't worry about other values
        # saving values
        minimizing_tau_save[0,iter] = tau.item()
        # CONCERN: nonsmooth jumps in contact functions
    else:
        # raise error
        # this error might be raised when the hoop is horizontal and centered at the hip, in which case there is no contact between hoop and hip and code proceeds normally
        raise NoLocalMinima()

    return gN, gNdot, gNddot, WN, gammaF, gammadotF, WF

def get_R(x, prev_x, prev_AV, prev_gammaF, prev_gNdot, prev_q, prev_u, *index_sets):
    # global iter

    # data extraction
    prev_a, _, _, _, _, _, _, _, _, _, prev_lambdaN, _, prev_lambdaF = \
        get_x_components(prev_x)
    a, U, Q, Kappa_g, Lambda_g, lambda_g, Lambda_gamma, lambda_gamma,\
        KappaN, LambdaN, lambdaN, LambdaF, lambdaF = get_x_components(x)
    
    # getting previous auxiliary variables
    prev_abar = prev_AV[0 : ndof]
    prev_lambdaNbar = prev_AV[ndof : ndof+nN]
    prev_lambdaFbar = prev_AV[ndof+nN : ndof+nN+nF]

    # calculating new auxiliary variables
    abar = (alpha_f*prev_a+(1-alpha_f)*a - alpha_m*prev_abar)/(1-alpha_m)  # (71)
    lambdaNbar = (alpha_f*prev_lambdaN+(1-alpha_f)*lambdaN
                    - alpha_m*prev_lambdaNbar)/(1-alpha_m)  # (96)
    lambdaFbar = (alpha_f*prev_lambdaF+(1-alpha_f)*lambdaF
                    - alpha_m*prev_lambdaFbar)/(1-alpha_m)  # (114)
    AV = np.concatenate((abar, lambdaNbar, lambdaFbar), axis=None)

    # Calculate q and u (73)
    u = prev_u+dtime*((1-gama)*prev_abar+gama*abar)+U
    q = prev_q+dtime*prev_u+dtime**2/2*((1-2*beta)*prev_abar+2*beta*abar)+Q

    # bilateral constraints at position level
    g = np.zeros((ng))
    gdot = np.zeros((ng))
    gddot = np.zeros((ng))
    Wg = np.zeros((ndof,ng))

    # bilateral constraints at velocity level
    gamma = np.zeros((ngamma))
    gammadot = np.zeros((ngamma))
    Wgamma = np.zeros((ndof,ngamma))

    # g[0] = q[0]
    # g[1] = q[1]
    # g[2] = q[2]

    # gdot[0] = u[0]
    # gdot[1] = u[1]
    # gdot[2] = u[2]

    # gddot[0] = a[0]
    # gddot[1] = a[1]
    # gddot[2] = a[2]

    # Wg[0,0] = 1
    # Wg[1,1] = 1
    # Wg[2,2] = 1

    # normal gap distance constraints and some frictional quantities
    gN, gNdot, gNddot, WN, gammaF, gammadotF, WF = combine_contact_constraints(q,u,a)

    # Kinetic quantities
    # normal
    ksiN = gNdot+eN*prev_gNdot # (86)
    PN = LambdaN+dtime*((1-gama)*prev_lambdaNbar+gama*lambdaNbar) # (95)
    Kappa_hatN = KappaN+dtime**2/2*((1-2*beta)*prev_lambdaNbar+2*beta*lambdaNbar) # (102)
    # frictional
    ksiF = gammaF+eF*prev_gammaF
    PF = LambdaF+dtime*((1-gama)*prev_lambdaFbar+gama*lambdaFbar) # (113)

    # Smooth residual Rs
    temp1 = M@a-f-np.transpose(Wg)@lambda_g-np.transpose(Wgamma)@lambda_gamma-np.transpose(WN)@lambdaN-np.transpose(WF)@lambdaF
    temp2 = M@U-np.transpose(Wg)@Lambda_g-np.transpose(Wgamma)@Lambda_gamma-np.transpose(WN)@LambdaN-np.transpose(WF)@LambdaF
    temp3 = M@Q-np.transpose(WN)@KappaN-np.transpose(Wg)@Kappa_g-dtime/2*np.transpose(WF)@LambdaF-dtime/2*np.transpose(Wgamma)@Lambda_gamma
    Rs = np.concatenate((temp1,temp2,temp3,g,gdot,gddot,gamma,gammadot),axis=None)

    # Contact residual Rc
    R_KappaN = np.zeros(nN)   # (129)
    R_LambdaN = np.zeros(nN)
    R_lambdaN = np.zeros(nN)
    R_LambdaF = np.zeros(nF)  # (138)
    R_lambdaF = np.zeros(nF)  # (142)

    

    if index_sets == ():
        A = np.zeros(nN, dtype=int)
        B = np.zeros(nN, dtype=int)
        C = np.zeros(nN, dtype=int)
        D = np.zeros(nF, dtype=int)
        E = np.zeros(nF, dtype=int)

        for i in range(nN):
            # check for contact if blocks are not horizontally detached
            if r*gN[i] - Kappa_hatN[i] <=0:
                A[i] = 1
                if np.linalg.norm(r*ksiF[gammaF_lim[i,:]]-PF[gammaF_lim[i,:]])<=mu_s*(PN[i]):
                    # D-stick
                    D[gammaF_lim[i,:]] = [1,1]
                    if np.linalg.norm(r*gammadotF[gammaF_lim[i,:]]-lambdaF[gammaF_lim[i,:]])<=mu_s*(lambdaN[i]):
                        # E-stick
                        E[gammaF_lim[i,:]] = [1,1]
                if r*ksiN[i]-PN[i] <= 0:
                    B[i] = 1
                    if r*gNddot[i]-lambdaN[i] <= 0:
                        C[i] = 1
    else:
        A = index_sets[0]
        B = index_sets[1]
        C = index_sets[2]
        D = index_sets[3]
        E = index_sets[4]

    # calculating contact residual
    for k in range(nN):
        if A[k]:
            R_KappaN[k] = gN[k]
            if D[k]:
                R_LambdaF[gammaF_lim[k,:]] = ksiF[gammaF_lim[k,:]]
                if E[k]:
                    R_lambdaF[gammaF_lim[k,:]] = gammadotF[gammaF_lim[k,:]]
                else:
                    R_lambdaF[gammaF_lim[k,:]] = lambdaF[gammaF_lim[k,:]]+mu_k*lambdaN[k]*np.sign(gammadotF[gammaF_lim[k,:]])                    
            else:
                R_LambdaF[gammaF_lim[k,:]] = PF[gammaF_lim[k,:]]+mu_k*PN[k]*np.sign(ksiF[gammaF_lim[k,:]])
                R_lambdaF[gammaF_lim[k,:]] = lambdaF[gammaF_lim[k,:]]+mu_k*lambdaN[k]*np.sign(gammaF[gammaF_lim[k,:]])
        else:
            R_KappaN[k] = Kappa_hatN[k]
            R_LambdaF[gammaF_lim[k,:]] = PF[gammaF_lim[k,:]]
            R_lambdaF[gammaF_lim[k,:]] = lambdaF[gammaF_lim[k,:]]
        # (132)
        if B[k]:
            R_LambdaN[k] = ksiN[k]
        else:
            R_LambdaN[k] = PN[k]
        # (135)
        if C[k]:
            R_lambdaN[k] = gNddot[k]
        else:
            R_lambdaN[k] = lambdaN[k]

    Rc = np.concatenate((R_KappaN,R_LambdaN,R_lambdaN,R_LambdaF,R_lambdaF),axis=None)

    # Assembling residual array
    Res = np.concatenate((Rs,Rc),axis=None)

    if index_sets == ():
        norm_R = np.linalg.norm(Res,np.inf)
        print(f'norm_R = {norm_R}')
        print(f'gN = [{gN[0]}, {gN[1]}]')
        gN_save[:,iter] = gN
        print(f'A = {A}')
        print(f'B = {B}')
        print(f'C = {C}')
        print(f'D = {D}')
        print(f'E = {E}')
        return Res, AV, gNdot, gammaF, q, u, A, B, C, D, E
    else:
        return Res, AV, gNdot, gammaF, q, u

def get_R_J(x,prev_x,prev_AV,prev_gammaF,prev_gNdot,prev_q,prev_u,leaf,*fixed_contact):
    
    epsilon = 1e-6
    fixed_contact_regions = False

    if fixed_contact != ():
        # here, the contact is fixed if a solve_bifurcation is being run
        fixed_contact = fixed_contact[0]
        fixed_contact_regions = True
        A = fixed_contact[0:nN]
        B = fixed_contact[nN:2*nN]
        C = fixed_contact[2*nN:3*nN]
        D = fixed_contact[3*nN:3*nN+nF]
        E = fixed_contact[3*nN+nF:3*nN+2*nF]
        R_x, AV, gNdot, gammaF, q, u =  get_R(x,prev_x,prev_AV,prev_gammaF,prev_gNdot,prev_q,prev_u,A, B, C, D, E)
    else:
        R_x, AV, gNdot, gammaF, q, u, A, B, C, D, E = get_R(x,prev_x,prev_AV,prev_gammaF,
                                            prev_gNdot,prev_q,prev_u)
        contacts_nu = np.concatenate((A,B,C,D,E),axis=None)
    
    # Initializing the Jacobian
    J = np.zeros((nX,nX))
    I = np.identity(nX)

    # Constructing the Jacobian column by column
    for i in range(nX):
        # print(i)
        R_x_plus_epsilon,_,_,_,_,_ = get_R(x+epsilon*I[:,i],prev_x,prev_AV,
                                           prev_gammaF,prev_gNdot,prev_q,prev_u,A, B, C, D, E)
        J[:,i] = (R_x_plus_epsilon-R_x)/epsilon

    if fixed_contact_regions:
        return R_x, AV, gNdot, gammaF, q, u, J
    else:
        # return the contact regions 'contacts_nu' to be saved in case they are needed (in the case of unconverged iterations)
        return R_x, AV, gNdot, gammaF, q, u, J, contacts_nu

## Solution
x_temp = x0
prev_x = x0

q[0,:] = q0
u[0,:] = u0
gammaF_save[0,:] = gammaF0
gNdot_save[0,:] = gNdot0

def update(prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,leaf,*fixed_contact):
    """Takes components at time t and return values at time t+dt"""
    global ntime, iter
    
    nu = 0
    X = prev_X
    
    if fixed_contact != ():
        # the contact region is fixed if solve_bifuration is calling update 
        # the fixed_contact data is inputted into get_R_J
        fixed_contact = fixed_contact[0]
        fixed_contact_regions = True
    else:
        fixed_contact_regions = False

    try:
        if fixed_contact_regions == True:
            R, AV, q, u, gNdot, gammaF, J = get_R_J(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,leaf,fixed_contact)
        else:
            R, AV, q, u, gNdot, gammaF, J, contacts_nu = get_R_J(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,leaf)
            contacts = np.zeros((MAXITERn+1,3*nN+2*nF),dtype=int)
            contacts[nu,:] = contacts_nu
        norm_R = np.linalg.norm(R,np.inf)
        # print(f"nu = {nu}")
        # print(f"norm(R) = {norm_R}")

        while np.abs(np.linalg.norm(R,np.inf))>(10**(-10)) and nu<MAXITERn:
            # Newton Update
            X = X-np.linalg.solve(J,R)
            # Calculate new EOM and residual
            nu = nu+1
            if fixed_contact_regions:
                R, AV, q, u, gNdot, gammaF, J = get_R_J(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,leaf,fixed_contact)
            else:
                R, AV, q, u, gNdot, gammaF, J, contacts_nu = get_R_J(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,leaf)
                contacts[nu,:] = contacts_nu
            norm_R = np.linalg.norm(R,np.inf)
            # print(f"nu = {nu}")
            # print(f"norm(R) = {norm_R}")
        if nu == MAXITERn:
            print(f"Iteration {iter} and leaf {leaf}:")
            print(f"No Convergence for nu = {nu} at rho_inf = {rho_inf}\n")
            raise MaxNewtonIterAttainedError
        
        if reduce_ntime_if_fail == 1:   # if we ask to stop code after failure is detected
            if 4 in corners_save:       # if failure is detected
                f.write(f"    Failure was detected at iteration {iter}, so ntime changed from {ntime} to {iter}.\n")
                ntime = iter
                raise FailureDetected

    except MaxNewtonIterAttainedError as e:
        if fixed_contact_regions is False:
            # if unique contact regions were already determined, don't recalculate them
            unique_contacts = np.unique(contacts, axis=0)
            do_not_unpack = True    
            # because if the number of contact regions is 6 which is the original number
            # of outputs of update, each row of unique contacts will be assinged as an output variable
            return unique_contacts, do_not_unpack
        return 
    except np.linalg.LinAlgError as e:
        # the Jacobian matrix is singular, not invertable
        print(f"Error {e} at iteration {iter} and leaf {leaf}.")
        # increment rho_inf        
        update_rho_inf()
        # calling function recursively
        update(prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,leaf,fixed_contact)
    except Exception as e:
        # any other exception
        g.write(f"\n    The exception {e} was raised during the calculation of some leaf at iteration {iter}.\n")
        print(f"Error {e} at iteration {iter} and leaf {leaf}.\n")
        raise e
    
    return X,AV,q,u,gNdot,gammaF

def update(prev_x, prev_AV, x_guess=None):

    if x_guess is None:
        x_guess = prev_x

    nu = 0
    
    x_temp = x_guess
    Res,AV_temp,gNdot_temp,gammaF_temp,q_temp,u_temp,J\
         = get_R_J(x_temp,prev_x,prev_AV,gammaF_save[iter-1,:],gNdot_save[iter-1,:],
                   q[iter-1,:],u[iter-1,:])
    
    norm_R = np.linalg.norm(Res,np.inf)

    # Semismooth Newton iterations
    while norm_R>tol_n and nu<maxiter_n:
        global n_tau

        # Newton update
        x_temp = x_temp-np.linalg.solve(J,Res)
        # Calculate new EOM and residual
        nu = nu+1
        print(f'nu = {nu}')
        Res,AV_temp,gNdot_temp,gammaF_temp,q_temp,u_temp,J = \
            get_R_J(x_temp,prev_x,prev_AV,gammaF_save[iter-1,:],gNdot_save[iter-1,:],\
                    q[iter-1,:],u[iter-1,:])
        norm_R = np.linalg.norm(Res,np.inf)

        if nu == maxiter_n:
            print(f'Maximum number of Newton iterations is exceeded at iterations {iter}')
            if n_tau*tol_n < 1001:
                n_tau = n_tau*10
                prev_x, prev_AV = update(prev_x,prev_AV,x_temp)
            else:
                raise MaxNewtonIterAttainedError()
                # here, I need to check if there is no convergence because of changing contact regions
                # currently I am assuming that the code is not converging because the tau array is not refined enough
                # SHOULD I PUT SOMETHING TO BREAK OUT OF FUNCTION HERE?
    
        R_array[iter] = norm_R
            
    # Updating reusable results
    gammaF_save[iter,:] = gammaF_temp
    gNdot_save[iter,:] = gNdot_temp
    q[iter,:] = q_temp
    u[iter,:] = u_temp
    prev_x = x_temp
    prev_AV = AV_temp 
    
    return x_temp, AV_temp

x_save[:,0] = prev_x
for iter in range(1,ntime):
    print(f'i={iter}')

    if iter == 202:
        print(202)

    # First semismooth Newton calculation
    t = dtime*iter
    n_tau = int(1/tol_n)
    try:
        prev_x, prev_AV = update(prev_x,prev_AV)
        x_save[:,iter-1] = prev_x
        if iter%10 ==0:
            save_arrays()
    finally:
        save_arrays()
        

        



print('done')