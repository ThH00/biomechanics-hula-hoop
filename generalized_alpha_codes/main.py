# seeking to simulate the motion of a hula hoop
# step 1. 2D, flat, circular hoop and hip, no friction. Release hoop and let rest on hip.
# step 2. move hip.
# step 3. add friction.
# step 4. expand to 3D.

## Importing packages
import numpy as np

## Problem constants
gr = 9.81                       # m/s^2, gravitational acceleration

# fixed basis
E1 = np.array([1,0,0])
E2 = np.array([0,1,0])
E3 = np.array([0,0,1])

## Simulation parameters
ti = 0                  # s, initial time
ntime = 1000            # dimensionless, number of iterations
dtime = 2e-3            # s, time step duration
t_arr = np.arange(0, ntime*dtime, dtime)


# hip properties
Dx = 0.2*np.cos(5*t_arr)
Dy = 0.6*np.sin(5*t_arr)
xD = np.column_stack((Dx, Dy, np.zeros(ntime)))

vDx = -0.2*5*np.sin(5*t_arr)
vDy = 0.6*5*np.cos(5*t_arr)
vD = np.column_stack((vDx, vDy, np.zeros(ntime)))

aDx = -0.2*25*np.cos(5*t_arr)
aDy = -0.6*25*np.sin(5*t_arr)
aD = np.column_stack((aDx, aDy, np.zeros(ntime)))


# Dx = 0                          # coordinates of center of hip
# Dy = 0                          # initially located at the origin
# xD = np.array([Dx, Dy, 0])
# vD = np.array([0,0,0])          # velocity of center of hip, currently fixed
# aD = np.array([0,0,0])          # acceleration of center of hip, currently fixed

d = 0.3                         # m, radius of hip         
omega_hip = np.array([0,0,0])   # angular velocity of hoop
alpha_hip = np.array([0,0,0])   # angular acceleration of hip

# hoop properties
ndof = 3                # number of degrees of freedom
b = 0.6                 # m, radius of hoop
m = 0.2                 # kg, mass of hoop
Izz = m*b**2            # kg.m^2, rotational inertia of hoop

# restitution coefficients
eN = 0.5                # dimensionless, normal impact restitution coefficient
eF = 0                  # dimensionless, tangential impact restitution coefficient

# friction coefficients
mu_s = 0.5              # dimensionless, static friction coefficient
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
tol_n = 1.0e-8

## Kinetic quantites
ng = 0                  # number of position level constraints
nN = 1                  # number of no penetration contact constraints
ngamma = 0              # number of velocity level constraints
nF = 1                  # slip speed constraints/friction force

## Initialize arrays to save results

R_array = np.zeros(ntime)

gamma_F = np.zeros((ntime,nF))
gdot_N = np.zeros((ntime,nN))

q = np.zeros((ntime,ndof))
u = np.zeros((ntime,ndof))

# initial values
q0 = np.array([0, 0, 0])
u0 = np.array([0, -1, 1])

nX = 3*ndof+3*ng+2*ngamma+3*nN+2*nF
x0 = np.zeros(nX)

# initial auxiliary variables
a_bar0 = np.zeros(ndof)
lambdaN_bar0 = np.zeros(nN)
lambdaF_bar0 = np.zeros(nF)

prev_AV = np.concatenate((a_bar0, lambdaN_bar0, lambdaF_bar0), axis=None)

gamma_F0 = np.zeros(nF)
gdot_N0 = np.zeros(nN)

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

def get_hoop_hip_contact(q,u,a):
    xB = q[:2]          # coordinates of the mass center of the hoop
    vB = u[:2]          # velocity of the mass center of the hoop
    aB = a[:2]          # acceleration of the mass center of the hoop

    omega_hoop = np.array([0,0,u[2]])   # angular velocity of hoop
    alpha_hoop = np.array([0,0,a[2]])   # angular acceleration of hoop

    # making the vectors three-dimensional
    xB = np.append(xB, 0)
    vB = np.append(vB, 0)
    aB = np.append(aB, 0)

    xBD = xB-xD[iter,:]
    vBD = vB-vD[iter,:]
    aBD = aB-aD[iter,:]

    BD = np.linalg.norm(xBD)  # distance between center of hoop and hip
    
    gN = np.zeros(nN)
    gNdot = np.zeros(nN)
    gNddot = np.zeros(nN)

    gN[0] = b-d-BD
    gNdot[0] = -np.dot(xBD,vBD)/(BD**2)
    gNddot[0] = ((np.dot(xBD,aBD)+np.dot(vBD,vBD))*BD-(np.dot(xBD,vBD))**2/BD)/(BD**2)

    WN = np.zeros((nN,ndof))
    WN[0,0] = -xBD[0]/BD
    WN[0,1] = -xBD[1]/BD

    gammaF = np.zeros(nF)
    gammadotF = np.zeros(nF)

    # normal-tangent basis
    normal = xBD/BD
    tangent = np.cross(E3, normal)
    normal_dot = 1/(BD)*(vBD -xBD*np.dot(vBD,xBD)/BD**2)
    tangent_dot = np.cross(E3, normal_dot)

    xE = xD[iter,:]+d*normal
    xF = xB+b*normal

    vE = vD[iter,:]+np.cross(omega_hip,xE-xD[iter,:])
    vF = vB+np.cross(omega_hoop,xF-xB)

    v_slip = vF - vE
    gammaF[0] = np.dot(v_slip, tangent)

    aE = aD[iter,:]+np.cross(alpha_hip,xE-xD[iter,:])+np.cross(omega_hip,vE-vD[iter,:])
    aF = aB+np.cross(alpha_hoop,xF-xB)+np.cross(omega_hoop,vF-vB)
    a_slip = aF - aE
    gammadotF[0] = np.dot(a_slip, tangent)+np.dot(v_slip, tangent_dot)   
    
    WF = np.zeros((nF,ndof))
    WF[0,0] = tangent[0]
    WF[0,1] = tangent[1]

    return gN, gNdot, gNddot, WN, gammaF, gammadotF, WF

def sign_no_zero(x):
    return np.where(x >= 0, 1, -1)

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
    Mdiag = np.array([m, m, Izz])
    M = np.diag(Mdiag)

    # Vector of applied forces and moments
    f = np.zeros(3)

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

    gN, gNdot, gNddot, WN, gammaF, gammadotF, WF = get_hoop_hip_contact(q,u,a)

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

    gammaF_lim = np.array([[0]])

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
                    R_lambdaF[gammaF_lim[k,:]] = lambdaF[gammaF_lim[k,:]]+mu_k*lambdaN[k]*sign_no_zero(gammadotF[gammaF_lim[k,:]])                    
            else:
                R_LambdaF[gammaF_lim[k,:]] = PF[gammaF_lim[k,:]]+mu_k*PN[k]*sign_no_zero(ksiF[gammaF_lim[k,:]])
                R_lambdaF[gammaF_lim[k,:]] = lambdaF[gammaF_lim[k,:]]+mu_k*lambdaN[k]*sign_no_zero(gammaF[gammaF_lim[k,:]])
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
        return Res, AV, gNdot, gammaF, q, u, A, B, C, D, E
    else:
        return Res, AV, gNdot, gammaF, q, u

def get_R_J(x,prev_x,prev_AV,prev_gamma_F,prev_gdot_N,prev_q,prev_u):
    
    epsilon = 1e-6
    # add return of index sets here
    R_x, AV, gdot_N, gamma_F, q, u, A, B, C, D, E = get_R(x,prev_x,prev_AV,prev_gamma_F,
                                           prev_gdot_N,prev_q,prev_u)
    n = np.size(R_x) # Jacobian dimension
    # Initializing the Jacobian
    J = np.zeros((n,n))
    I = np.identity(n)
    # Constructing the Jacobian column by column
    for i in range(n):
        # print(i)
        R_x_plus_epsilon,_,_,_,_,_ = get_R(x+epsilon*I[:,i],prev_x,prev_AV,
                                           prev_gamma_F,prev_gdot_N,prev_q,prev_u,A, B, C, D, E)
        J[:,i] = (R_x_plus_epsilon-R_x)/epsilon

    return R_x,AV,gdot_N,gamma_F,q,u,J

## Solution
x_temp = x0
prev_x = x0

q[0,:] = q0
u[0,:] = u0
gamma_F[0,:] = gamma_F0
gdot_N[0,:] = gdot_N0

for iter in range(1,ntime):
    print(f'i={iter}')

    # First semismooth Newton calculation
    t = dtime*iter
    nu = 0

    Res,AV_temp,gdot_N_temp,gamma_F_temp,q_temp,u_temp,J\
         = get_R_J(x_temp,prev_x,prev_AV,gamma_F[iter-1,:],gdot_N[iter-1,:],
                   q[iter-1,:],u[iter-1,:])
    
    norm_R = np.linalg.norm(Res,np.inf)
    print(f'lambda_N = {x_temp[3*ndof+3*ng+2*ngamma+2*nN:3*ndof+3*ng+2*ngamma+3*nN]}')
    print(f'lambda_g = {x_temp[3*ndof+2*ng:3*ndof+3*ng]}')

    # Semismooth Newton iterations
    while norm_R>tol_n and nu<maxiter_n:
        # Newton update
        x_temp = x_temp-np.linalg.solve(J,Res)
        # Calculate new EOM and residual
        nu = nu+1
        print(f'nu = {nu}')
        Res,AV_temp,gdot_N_temp,gamma_F_temp,q_temp,u_temp,J = \
            get_R_J(x_temp,prev_x,prev_AV,gamma_F[iter-1,:],gdot_N[iter-1,:],\
                    q[iter-1,:],u[iter-1,:])
        norm_R = np.linalg.norm(Res,np.inf)
        print(f'norm_R = {norm_R}')
        print(f'lambda_N = {x_temp[3*ndof+3*ng+2*ngamma+2*nN:3*ndof+3*ng+2*ngamma+3*nN]}')
        print(f'lambda_g = {x_temp[3*ndof+2*ng:3*ndof+3*ng]}')
    
    R_array[iter] = norm_R
        
    # Updating reusable results
    gamma_F[iter,:] = gamma_F_temp
    gdot_N[iter,:] = gdot_N_temp
    q[iter,:] = q_temp
    u[iter,:] = u_temp
    prev_x = x_temp
    prev_AV = AV_temp 

import scipy.io

output_path = '/Users/theresahonein/Desktop/terryhonein/Research-HulaHoop'

file_name_q = str(f'{output_path}/q.mat')
file_name_u = str(f'{output_path}/u.mat')
scipy.io.savemat(file_name_q,dict(q=q))
scipy.io.savemat(file_name_u,dict(u=u))

file_name_J = str(f'{output_path}/J.mat')
scipy.io.savemat(file_name_J,dict(J=J))

file_name_J = str(f'{output_path}/xD.mat')
scipy.io.savemat(file_name_J,dict(xD=xD))

print('done')