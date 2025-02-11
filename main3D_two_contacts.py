# simulating the motion of a hula hoop

## Importing packages
import numpy as np
from scipy.optimize import minimize
import sympy as sp
from scipy.signal import argrelextrema

## Problem constants
gr = 9.81           # m/s^2, gravitational acceleration

# fixed basis
E1 = np.array([1,0,0])
E2 = np.array([0,1,0])
E3 = np.array([0,0,1])

## Simulation parameters
ti = 0              # s, initial time
ntime = 500        # dimensionless, number of iterations
dtime = 2e-3        # s, time step duration
t_arr = np.arange(0, ntime*dtime, dtime)

## Hip axis properties

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

# The hip center is fixed
R_hip = 0.2
# Position of the bottom center of hip (bottom of hip aixs)
xbar_hip = np.zeros((ntime,3))
# velocity of the bottom center of hip
vbar_hip = np.zeros((ntime,3))
# acceleration of the bottom center of hip
abar_hip = np.zeros((ntime,3))

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
eN = 0.5                # dimensionless, normal impact restitution coefficient
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
maxiter_n = 100
tol_n = 1.0e-6

## Kinetic quantites
ng = 0                  # number of position level constraints
nN = 2                  # number of no penetration contact constraints
ngamma = 0              # number of velocity level constraints
nF = 4                  # slip speed constraints/friction force

## Initialize arrays to save results

R_array = np.zeros(ntime)

gammaF = np.zeros((ntime,nF))
gdotN = np.zeros((ntime,nN))

q = np.zeros((ntime,ndof))
u = np.zeros((ntime,ndof))

# initial values
q0 = np.array([0.1, 0, 1, 0, np.pi/6, 0])
u0 = np.array([1, 0, 0, 0, 0, 0])

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

def get_hoop_hip_contact(q,u,a):

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
    e3dot = np.cross(omega_hoop, e3)

    alpha_hoop = psiddot*E3+thetaddot*e1p+thetadot*e1pdot+phiddot*e3+phidot*e3dot

    n_tau = int(1/tol_n)
    tau = np.linspace(0, 2*np.pi, num=n_tau, endpoint=False)
    # I can find intervals containing the minima and then refine the discretization in the particular interval (or use the bisection method)

    n = np.zeros((3,n_tau))
    nH = np.zeros((3,n_tau))
    dH = np.zeros(n_tau)

    for i in range(n_tau):
        # n is a vector along the radius of the hoop making an angle tau with e1
        # tau varies between [0,2*pi)
        n[:,i] = np.cos(tau[i])*e1+np.sin(tau[i])*e2
        # horizontal projection of n
        temp = n[:,i]-np.dot(n[:,i],E3)
        nH[:,i] = temp/np.linalg.norm(temp)
        # horinzontal distance
        dH[i] = np.dot(xbar_hoop+R_hoop*n-xbar_hip,nH)-R_hip

    # Find local minima (less than neighbors)
    min_indices = argrelextrema(dH, np.less)[0]

    # Extract maxima and minima values
    min_values = dH[min_indices]

    # Display results
    print("Local minima (index, value):", list(zip(min_indices, min_values)))

    # Find the minizing value of tau
    minimizing_tau = tau[min_indices]

    # Minimizing value of n
    minimizing_n = n[:,min_indices]

    # Number of minimizers
    num_minimizers = np.size(minimizing_tau)

    minimizing_x3 = np.zeros(num_minimizers)
    for i in range(num_minimizers):
        minimizing_x3[i] = np.dot(xbar_hoop + R_hoop * minimizing_n[:,i] - xbar_hip, E3)
    

    
    ###############################################


    gN = dH/np.cos(theta)

    # right handed orthonormal basis of contact point
    n = np.cos(tau)*e1+np.sin(tau)*e2
    t1 = -np.sin(tau)*e1+np.cos(tau)*e2
    t2 = np.cross(n,t1)

    xhoop = xbar_hoop+R_hoop*n
    R_hip = 0.2
    xhip = xbar_hip[iter,:]+R_hip*n+x3*E3 # this is not true. need to have a trigonometric function of theta somewhere

    gN = np.dot(xhoop-xhip,n)
    gNdot = -vbar_hip[iter,0]*np.cos(tau) - vbar_hip[iter,1]*np.sin(tau) + vbar_hoop[0]*np.cos(tau) + vbar_hoop[1]*np.sin(tau)
    gNddot = -abar_hip[iter,0]*np.cos(tau) - abar_hip[iter,1]*np.sin(tau) + abar_hoop[0]*np.cos(tau) + abar_hoop[1]*np.sin(tau)

    WN = np.zeros((nN,ndof))


    # Position vectors to points P and Q
    xQ = xbar_hoop+R_hoop*n
    xP = xbar_hip[iter,:]+R_hip*n

    # Velocities of points P and Q
    vQ = vbar_hoop+np.cross(omega_hoop,xQ-xbar_hoop)
    vP = vbar_hip[iter,:]+np.cross(omega_hip,xP-xbar_hip[iter,:])

    # Slip speeds
    gammaF1 = np.dot(vP-vQ,t1)
    gammaF2 = np.dot(vP-vQ,t2)

    gammaF = np.array([gammaF1, gammaF2])

    # Slip speed derivatives
    aQ = abar_hoop+np.cross(alpha_hoop,xQ-xbar_hoop)+np.cross(omega_hoop,vQ-vbar_hoop)
    aP = abar_hip[iter,:]+np.cross(alpha_hip,xP-xbar_hip[iter,:])+np.cross(omega_hip,vP-vbar_hip[iter,:])

    t1dot = np.cross(omega_hoop,t1)
    t2dot = np.cross(omega_hoop,t2)

    gammadotF1 = np.dot(aP-aQ,t1)+np.dot(vP-vQ,t1dot)
    gammadotF2 = np.dot(aP-aQ,t2)+np.dot(vP-vQ,t2dot)

    gammadotF = np.array([gammadotF1, gammadotF2])

    WF = np.zeros((nF, ndof))



    return gN, gNdot, gNddot, WN, gammaF, gammadotF, WF


def get_R(x, prev_x, prev_AV, prev_gammaF, prev_gdotN, prev_q, prev_u, *index_sets):
    global iter

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
    # f = np.array([0, 0, -m*gr, 0, 0, 0])      # I removed gravity for now
    f = np.array([0, 0, 0, 0, 0, 0])

    # Constraints and constraint gradients
    # initializing constraints
    g = np.zeros(ng)
    gamma = np.zeros(ngamma)
    gN = np.zeros(nN)
    gammaF = np.zeros(nF)
    # initialize constraint derivatives
    gdot = np.zeros(ng)
    gddot = np.zeros(ng)
    gammadot = np.zeros(ngamma)
    gNdot = np.zeros(nN)
    gNddot = np.zeros(nN)
    gammadotF = np.zeros(nF)
    # initializing constraint gradients
    Wg = np.zeros((ng, ndof))
    Wgamma = np.zeros((ngamma, ndof))
    WN = np.zeros((nN, ndof))
    WF = np.zeros((nF, ndof))

    gN[0], gNdot[0], gNddot[0], WN[0,:], gammaF, gammadotF, WF = get_hoop_hip_contact(q,u,a)

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

    gammaF_lim = np.array([[0,1]])

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
        print(f'gN = {gN[0]}')
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

for iter in range(1,ntime):
    print(f'i={iter}')

    # First semismooth Newton calculation
    t = dtime*iter
    nu = 0

    Res,AV_temp,gdot_N_temp,gammaF_temp,q_temp,u_temp,J\
         = get_R_J(x_temp,prev_x,prev_AV,gammaF[iter-1,:],gdotN[iter-1,:],
                   q[iter-1,:],u[iter-1,:])
    
    norm_R = np.linalg.norm(Res,np.inf)
    # print(f'lambda_N = {x_temp[3*ndof+3*ng+2*ngamma+2*nN:3*ndof+3*ng+2*ngamma+3*nN]}')
    # print(f'lambda_g = {x_temp[3*ndof+2*ng:3*ndof+3*ng]}')

    # Semismooth Newton iterations
    while norm_R>tol_n and nu<maxiter_n:
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
        # print(f'norm_R = {norm_R}')
        # print(f'lambda_N = {x_temp[3*ndof+3*ng+2*ngamma+2*nN:3*ndof+3*ng+2*ngamma+3*nN]}')
        # print(f'lambda_g = {x_temp[3*ndof+2*ng:3*ndof+3*ng]}')
    
    R_array[iter] = norm_R
        
    # Updating reusable results
    gammaF[iter,:] = gammaF_temp
    gdotN[iter,:] = gdot_N_temp
    q[iter,:] = q_temp
    u[iter,:] = u_temp
    prev_x = x_temp
    prev_AV = AV_temp 

import scipy.io

output_path = '/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop'
file_name_J = str(f'{output_path}/J.mat')
scipy.io.savemat(file_name_J,dict(J=J))

file_name_q = str(f'{output_path}/q.mat')
file_name_u = str(f'{output_path}/u.mat')
scipy.io.savemat(file_name_q,dict(q=q))
scipy.io.savemat(file_name_u,dict(u=u))

file_name_gN = str(f'{output_path}/gN.mat')
scipy.io.savemat(file_name_gN,dict(gN=gN_save))

file_name_xbar_hip = str(f'{output_path}/xbar_hip.mat')
scipy.io.savemat(file_name_xbar_hip,dict(xbar_hip=xbar_hip))

print('done')