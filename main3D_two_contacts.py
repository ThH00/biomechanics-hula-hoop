# simulating the motion of a hula hoop

## Importing packages
import numpy as np
import sympy as sp
from scipy.signal import argrelextrema
import scipy.io

# Define a custom error
class NoLocalMinima(Exception):
    def __init__(self, message="The distance between the hoop and the hip has less then 1 or more than 2 local minima."):
        super().__init__(message)

class MaxNewtonIterAttainedError(Exception):
    """This exception is raised when the maximum number of Newton iterations is attained
      whilst the iterations have not yet converged and the solution was not yet obtained."""
    def __init__(self, message="This exception is raised when the maximum number of Newton iterations is attained."):
        self.message = message
        super().__init__(self.message)

## Problem constants
gr = 9.81           # m/s^2, gravitational acceleration

# fixed basis
E1 = np.array([1,0,0])
E2 = np.array([0,1,0])
E3 = np.array([0,0,1])

## Simulation parameters
ti = 0              # s, initial time
ntime = 2000        # dimensionless, number of iterations
dtime = 2e-3        # s, time step duration
t_arr = np.arange(0, ntime*dtime, dtime)

## Hip axis properties
R_hip = 0.2

# The hip center is tracing an ellipse
# Position of the bottom center of hip (bottom of hip axis)
x1bar_hip = 0.2*np.cos(5*t_arr)
x2bar_hip = 0.6*np.sin(5*t_arr)
xbar_hip = np.column_stack((x1bar_hip, x2bar_hip, np.zeros(ntime)))
# velocity of the bottom center of hip
v1bar_hip = -0.2*5*np.sin(5*t_arr)
v2bar_hip = 0.6*5*np.cos(5*t_arr)
vbar_hip = np.column_stack((v1bar_hip, v2bar_hip, np.zeros(ntime)))
# acceleration of the bottom center of hip
a1bar_hip = -0.2*25*np.cos(5*t_arr)
a2bar_hip = -0.6*25*np.sin(5*t_arr)
abar_hip = np.column_stack((a1bar_hip, a2bar_hip, np.zeros(ntime)))

# # The hip center is fixed
# # Position of the bottom center of hip (bottom of hip aixs)
# xbar_hip = np.zeros((ntime,3))
# # velocity of the bottom center of hip
# vbar_hip = np.zeros((ntime,3))
# # acceleration of the bottom center of hip
# abar_hip = np.zeros((ntime,3))

# Angular velocity and angular acceleration of hip
# omega_hip = np.array([0,0,1])   # angular velocity of hip
omega_hip = np.array([0,0,0])   # angular velocity of hip
alpha_hip = np.array([0,0,0])   # angular acceleration of hip

# hoop properties
ndof = 6                # number of degrees of freedom
R_hoop = 0.5            # m, radius of hoop
m = 0.2                 # kg, mass of hoop
It = 0.5*m*R_hoop**2    # kg.m^2, rotational inertia of hoop about diameter
Ia = m*R_hoop**2        # kg.m^2, rotational inertia of hoop about axis passing through center perp to hoop plane

# restitution coefficients
eN = 0                # dimensionless, normal impact restitution coefficient
eF = 0                  # dimensionless, tangential impact restitution coefficient

# friction coefficients
mu_s = 0.8              # dimensionless, static friction coefficient
mu_k = 0.2              # dimensionless, kinetic friction coefficient


## Parameters of the generalized-alpha scheme
# differentiation parameters
rho_inf = 0.5
alpha_m = (2*rho_inf-1)/(rho_inf+1)
alpha_f = rho_inf/(rho_inf+1)
gama = 0.5+alpha_f-alpha_m
beta = 0.25*(0.5+gama)**2

# set approximation parameter
r = 0.3

# loop parameters
maxiter_n = 20
tol_n = 1.0e-6

## Kinetic quantites
ng = 0                  # number of position level constraints
nN = 2                  # number of no penetration contact constraints
ngamma = 0              # number of velocity level constraints
nF = 4                  # slip speed constraints/friction force
gammaF_lim = np.array([[0,1],[2,3]])

## Initialize arrays to save results

R_array = np.zeros(ntime)

gammaF = np.zeros((ntime,nF))
gdotN = np.zeros((ntime,nN))

q = np.zeros((ntime,ndof))
u = np.zeros((ntime,ndof))

# initial values
q0 = np.array([0.05, 0.1, 1, 0, np.pi/6, 0])
u0 = np.array([1, 0, 0, 0, 3, 0])

nX = 3*ndof+3*ng+2*ngamma+3*nN+2*nF
x0 = np.zeros(nX)

# initial auxiliary variables
a_bar0 = np.zeros(ndof)
lambdaN_bar0 = np.zeros(nN)
lambdaF_bar0 = np.zeros(nF)

prev_AV = np.concatenate((a_bar0, lambdaN_bar0, lambdaF_bar0), axis=None)

gammaF0 = np.zeros(nF)
gdotN0 = np.zeros(nN)

gN_save = np.zeros((nN, ntime))
x_save = np.zeros((3*ndof+3*ng+2*ngamma+3*nN+2*nF, ntime))
minimizing_tau_save = np.zeros((2,ntime))

output_path = '/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop'
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

    gN = -R_hip + (xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    gNdot = phidot*(1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + psidot*(1.0*R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + thetadot*(1.0*R_hoop*(np.sin(phi)*np.sin(psi)*np.sin(theta)*np.cos(tau) + np.sin(psi)*np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*(-np.sin(phi)*np.sin(theta)*np.cos(psi)*np.cos(tau) - np.sin(tau)*np.sin(theta)*np.cos(phi)*np.cos(psi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + vbar_hip[0]*(-1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) + 1.0*xbar_hip[0] - 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + vbar_hip[1]*(-1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + 1.0*xbar_hip[1] - 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + 1.0*vbar_hip[2]*xbar_hip[2]/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + vbar_hoop[0]*(1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - 1.0*xbar_hip[0] + 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + vbar_hoop[1]*(1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - 1.0*xbar_hip[1] + 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    gNddot = 1.0*R_hoop*phidot**2*(-R_hoop*(((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.sin(tau))*(-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]) + ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) - (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))*(-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]))**2/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.sin(tau))**2 + R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) - (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))**2 - ((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1]) - ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**0.5) + 1.0*R_hoop*psidot**2*(-R_hoop*(((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) - ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))*(-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))*(-((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0]) + ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))**2 + R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))**2 - ((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1]) - ((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**0.5) + 1.0*R_hoop*thetadot**2*(np.sin(phi)*np.cos(tau) + np.sin(tau)*np.cos(phi))*(-R_hoop*(-(-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*np.cos(psi) + (-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*np.sin(psi))*((R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])*np.cos(psi) - (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])*np.sin(psi))*(np.sin(phi)*np.cos(tau) + np.sin(tau)*np.cos(phi))*np.sin(theta)**2/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (R_hoop*(np.sin(phi)*np.cos(tau) + np.sin(tau)*np.cos(phi))*np.sin(psi)**2*np.sin(theta)**2 + R_hoop*(np.sin(phi)*np.cos(tau) + np.sin(tau)*np.cos(phi))*np.sin(theta)**2*np.cos(psi)**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])*np.cos(psi)*np.cos(theta) - (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])*np.sin(psi)*np.cos(theta))/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**0.5) + abar_hip[0]*(-1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) + 1.0*xbar_hip[0] - 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hip[1]*(-1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + 1.0*xbar_hip[1] - 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + 1.0*abar_hip[2]*xbar_hip[2]/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hoop[0]*(1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - 1.0*xbar_hip[0] + 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hoop[1]*(1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - 1.0*xbar_hip[1] + 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + phiddot*(1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + psiddot*(1.0*R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + thetaddot*(1.0*R_hoop*(np.sin(phi)*np.sin(psi)*np.sin(theta)*np.cos(tau) + np.sin(psi)*np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*(-np.sin(phi)*np.sin(theta)*np.cos(psi)*np.cos(tau) - np.sin(tau)*np.sin(theta)*np.cos(phi)*np.cos(psi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + 1.0*vbar_hip[0]**2*((-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5)) + 1.0*vbar_hip[1]**2*((-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5)) + 1.0*vbar_hip[2]**2*(-xbar_hip[2]**2/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5)) + 1.0*vbar_hoop[0]**2*((-R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*(R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5)) + 1.0*vbar_hoop[1]**2*((-R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*(R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**1.5 + (xbar_hip[2]**2 + (R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) - (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + xbar_hip[1] - xbar_hoop[1])**2 + (R_hoop*((np.sin(phi)*np.cos(psi) + np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.cos(tau)) + xbar_hip[0] - xbar_hoop[0])**2)**(-0.5))

    WN[0,0] = (1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - 1.0*xbar_hip[0] + 1.0*xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    WN[0,1] = (1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - 1.0*xbar_hip[1] + 1.0*xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    WN[0,2] = 0
    WN[0,3] = (1.0*R_hoop*((np.sin(phi)*np.sin(psi) - np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    WN[0,4] = (1.0*R_hoop*(np.sin(phi)*np.sin(psi)*np.sin(theta)*np.cos(tau) + np.sin(psi)*np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]) + 1.0*R_hoop*(-np.sin(phi)*np.sin(theta)*np.cos(psi)*np.cos(tau) - np.sin(tau)*np.sin(theta)*np.cos(phi)*np.cos(psi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    WN[0,5] = (1.0*R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.cos(tau) + (-np.sin(phi)*np.cos(psi)*np.cos(theta) - np.sin(psi)*np.cos(phi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1]) + 1.0*R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.cos(tau) + (np.sin(phi)*np.sin(psi)*np.cos(theta) - np.cos(phi)*np.cos(psi))*np.sin(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0]))/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5

    gammaF1 = R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(phidot*np.sin(psi)*np.sin(theta) + thetadot*np.cos(psi)) - R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(-phidot*np.sin(theta)*np.cos(psi) + thetadot*np.sin(psi)) - vbar_hip[2] + vbar_hoop[2]
    gammaF2 = (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*(-R_hoop*(phidot*np.cos(theta) + psidot)*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + R_hoop*(-phidot*np.sin(theta)*np.cos(psi) + thetadot*np.sin(psi))*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) - vbar_hip[0] + vbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*(R_hoop*(phidot*np.cos(theta) + psidot)*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - R_hoop*(phidot*np.sin(psi)*np.sin(theta) + thetadot*np.cos(psi))*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) - vbar_hip[1] + vbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
    
    gammadotF1 = -abar_hip[2] + abar_hoop[2] + phiddot*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.sin(psi)*np.sin(theta) + R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(theta)*np.cos(psi)) + thetaddot*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(psi) - R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(psi))
    gammadotF2 = -abar_hip[0]*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hip[1]*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hoop[0]*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - abar_hoop[1]*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + phiddot*((-R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(theta) - R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(theta)*np.cos(psi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.cos(theta) - R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(psi)*np.sin(theta))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5) + psiddot*(-R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5) + thetaddot*(R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*np.sin(psi)/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*np.cos(psi)/(xbar_hip[2]**2 + (R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5)

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


def get_R(x, prev_x, prev_AV, prev_gammaF, prev_gdotN, prev_q, prev_u, *index_sets):
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

    # Mass matrix
    Mdiag = np.array([m, m, m, It, It, Ia])
    M = np.diag(Mdiag)

    # Vector of applied forces and moments
    f = np.array([0, 0, -m*gr, 0, 0, 0])      # I removed gravity for now
    # f = np.array([0, 0, 0, 0, 0, 0])

    # Constraints and constraint gradients
    # initializing constraints
    g = np.zeros(ng)
    gamma = np.zeros(ngamma)
    # initialize constraint derivatives
    gdot = np.zeros(ng)
    gddot = np.zeros(ng)
    gammadot = np.zeros(ngamma)
    # initializing constraint gradients
    Wg = np.zeros((ng, ndof))
    Wgamma = np.zeros((ngamma, ndof))

    gN, gNdot, gNddot, WN, gammaF, gammadotF, WF = combine_contact_constraints(q,u,a)

    # Kinetic quantities
    # normal
    ksiN = gNdot+eN*prev_gdotN # (86)
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
                if np.abs(r*ksiF[i]-PF[i])<=mu_s*(PN[i]):
                    # D-stick
                    D[i] = 1
                    if np.abs(r*gammadotF[i]-lambdaF[i])<=mu_s*(lambdaN[i]):
                        # E-stick
                        E[i] = 1
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
        gN_save[0,iter] = gN[0]
        return Res, AV, gNdot, gammaF, q, u, A, B, C, D, E
    else:
        return Res, AV, gNdot, gammaF, q, u

def get_R_J(x,prev_x,prev_AV,prev_gammaF,prev_gdot_N,prev_q,prev_u):
    
    epsilon = 1e-6
    # add return of index sets here
    R_x, AV, gdot_N, gammaF, q, u, A, B, C, D, E = get_R(x,prev_x,prev_AV,prev_gammaF,
                                           prev_gdot_N,prev_q,prev_u)
    n = np.size(R_x) # Jacobian dimension
    # Initializing the Jacobian
    J = np.zeros((n,n))
    I = np.identity(n)
    # Constructing the Jacobian column by column
    for i in range(n):
        # print(i)
        R_x_plus_epsilon,_,_,_,_,_ = get_R(x+epsilon*I[:,i],prev_x,prev_AV,
                                           prev_gammaF,prev_gdot_N,prev_q,prev_u,A, B, C, D, E)
        J[:,i] = (R_x_plus_epsilon-R_x)/epsilon

    return R_x,AV,gdot_N,gammaF,q,u,J

## Solution
x_temp = x0
prev_x = x0

q[0,:] = q0
u[0,:] = u0
gammaF[0,:] = gammaF0
gdotN[0,:] = gdotN0

def update(prev_x,prev_AV):
    nu = 0
    
    x_temp = prev_x
    Res,AV_temp,gdot_N_temp,gammaF_temp,q_temp,u_temp,J\
         = get_R_J(x_temp,prev_x,prev_AV,gammaF[iter-1,:],gdotN[iter-1,:],
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
        Res,AV_temp,gdot_N_temp,gammaF_temp,q_temp,u_temp,J = \
            get_R_J(x_temp,prev_x,prev_AV,gammaF[iter-1,:],gdotN[iter-1,:],\
                    q[iter-1,:],u[iter-1,:])
        norm_R = np.linalg.norm(Res,np.inf)

        if nu == maxiter_n:
            print(f'Maximum number of Newton iterations is exceeded at iterations {iter}')
            if n_tau*tol_n < 1001:
                n_tau = n_tau*10
                prev_x, prev_AV = update(prev_x,prev_AV)
            else:
                raise MaxNewtonIterAttainedError()
                # here, I need to check if there is no convergence because of changing contact regions
                # currently I am assuming that the code is not converging because the tau array is not refined enough
                # SHOULD I PUT SOMETHING TO BREAK OUT OF FUNCTION HERE?
    
        R_array[iter] = norm_R
            
    # Updating reusable results
    gammaF[iter,:] = gammaF_temp
    gdotN[iter,:] = gdot_N_temp
    q[iter,:] = q_temp
    u[iter,:] = u_temp
    prev_x = x_temp
    prev_AV = AV_temp 
    
    return x_temp, AV_temp

x_save[:,0] = prev_x
for iter in range(1,ntime):
    print(f'i={iter}')

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