# I will remove time tracking for now.

import numpy as np
import time
import os
import argparse
from scipy.signal import argrelextrema
import scipy.io
from datetime import datetime
import shutil


# creating custom exceptions
class MaxNewtonIterAttainedError(Exception):
    """This exception is raised when the maximum number of Newton iterations is attained
      whilst the iterations have not yet converged and the solution was not yet obtained."""
    def __init__(self, message="This exception is raised when the maximum number of Newton iterations is attained."):
        self.message = message
        super().__init__(self.message)

class NoOpenContactError(Exception):
    """Contact is not open."""
    def __init__(self, message="This exception is raised when the contact is not open."):
        self.message = message
        super().__init__(self.message)

class RhoInfInfiniteLoop(Exception):
    """This exception is raised when we have possibly entered in an infinite loop through updating rho_inf."""
    def __init__(self, message="This exception is raised when we have possibly entered in an infinite loop through updating rho_inf."):
        self.message = message
        super().__init__(self.message)

class MaxHoursAttained(Exception):
    """This exception is raised when the maximum number of run hours specified by the user is exceeded."""
    def __init__(self, message="This exception is raised when the maximum run time is exceeded."):
        self.message = message
        super().__init__(self.message)

class MaxLeavesAttained(Exception):
    """This exception is raised when the maximum number of leaves specified by the user is exceeded."""
    def __init__(self, message="This exception is raised when the maximum number of leaves is exceeded."):
        self.message = message
        super().__init__(self.message)

class NoBifurcationConvergence(Exception):
    """This exception is raised when none of the leaves converged."""
    def __init__(self, message="This exception is raised when none of the leaves converged."):
        self.message = message
        super().__init__(self.message)

class JacobianBlowingUpError(Exception):
    """This exception is raised when the Jacobian is blowing up."""
    def __init__(self, message="This exception is raised when the Jacobian is blowing up."):
        self.message = message
        super().__init__(self.message)

class Simulation:
    def __init__(self, ntime = 5, mu_s=10**9, mu_k=0.3, eN=0, eF=0, max_leaves=5):
        # path for outputs
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outputs_dir = f"outputs/{timestamp}"
        self.output_path = os.path.join(os.getcwd(), outputs_dir)  # Output path
        os.makedirs(self.output_path, exist_ok=True)

        # Path to the current file
        current_file = os.path.realpath(__file__)
        # Copy the file
        shutil.copy2(current_file, self.output_path)

        # friction coefficients
        self.mu_s = mu_s    # Static friction coefficient
        self.mu_k = mu_k    # Kinetic friction coefficient
        # restitution coefficients
        self.eN = eN        # normal coefficient of restitution
        self.eF = eF        # friction coefficient of restitution
        # nondimensionalization parameters
        l_nd = 1       # m, length nondimensionalization paramter
        m_nd = 1       # kg, mass nondimensionalization parameter
        a_nd = 9.81    # m/(s**2), acceleration nondimensionalization parameter
        t_nd = np.sqrt(l_nd/a_nd)   # s, time nondimensionalization parameter
        # simulation (time) parameters
        self.dtime = 2e-3/t_nd # time step duration
        self.ntime = ntime           # number of iterations
        self.tf = self.ntime*self.dtime            # final time
        self.t = np.linspace(0,self.tf,self.ntime) # time array
        # hip properties
        self.R_hip = 0.2/l_nd            # radius of the hip, hip is circular

        # hip motion
        # the hip is tracing an ellipse with angular velocity omega, major axis a, and minor axis b
        # omega = 1
        # a = 1
        # b = 1
        # x1bar_hip = a*np.cos(omega*self.t)
        # x2bar_hip = b*np.sin(omega*self.t)
        # self.xbar_hip = np.column_stack((x1bar_hip, x2bar_hip, np.zeros(ntime)))
        # v1bar_hip = a*omega*np.cos(omega*self.t)
        # v2bar_hip = b*omega*np.sin(omega*self.t)
        # self.vbar_hip = np.column_stack((v1bar_hip, v2bar_hip, np.zeros(ntime)))
        # a1bar_hip = a*omega**2*np.cos(omega*self.t)
        # a2bar_hip = b*omega**2*np.sin(omega*self.t)
        # self.abar_hip = np.column_stack((a1bar_hip, a2bar_hip, np.zeros(ntime)))
        xbar_hip = np.zeros((self.ntime,3))
        vbar_hip = np.zeros((self.ntime,3))
        abar_hip = np.zeros((self.ntime,3))

        # psi_hip = np.zeros((self.ntime,1))
        # theta_hip = np.zeros((self.ntime,1))
        psi_hip = (np.pi/2)*np.ones((self.ntime,1))
        theta_hip = (np.pi/2)*np.ones((self.ntime,1))
        phi_hip = np.zeros((self.ntime,1))

        psidot_hip = np.zeros((self.ntime,1))
        thetadot_hip = np.zeros((self.ntime,1))
        phidot_hip = np.zeros((self.ntime,1))

        psiddot_hip = np.zeros((self.ntime,1))
        thetaddot_hip = np.zeros((self.ntime,1))
        phiddot_hip = np.zeros((self.ntime,1))

        # CHECK HERE
        self.q_hip = np.concatenate((xbar_hip, psi_hip, theta_hip, phi_hip),axis=1)
        self.u_hip = np.concatenate((vbar_hip, psidot_hip, thetadot_hip, phidot_hip),axis=1)
        self.a_hip = np.concatenate((abar_hip, psiddot_hip, thetaddot_hip, phiddot_hip),axis=1)

        # hoop properties
        self.m = 0.2/m_nd      # mass of hoop
        self.R_hoop = 0.5/l_nd           # radius of hoop
        self.It = 0.5*self.m*self.R_hoop**2   # rotational inertia of hoop about diameter
        self.Ia = self.m*self.R_hoop**2       # rotational inertia of hoop about axis passing through center perp to hoop plane
        # nondimensional constants
        self.ndof = 6               # total number of degress of freedom
        self.gr = 9.81/a_nd    # gravitational acceleration
        # constraint count
        self.ng = 0          # number of constraints at position level
        self.ngamma = 0      # number of constraints at velocity level
        self.nN = 2          # number of gap distance constraints
        self.nF = 4          # number of friction constraints
        self.gammaF_lim = np.array([[0,1],[2,3]])    # connectivities of friction and normal forces
        self.nX = 3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+2*self.nF     # total number of constraints with their derivative
        # fixed basis vectors
        self.E1 = np.array([1,0,0])
        self.E2 = np.array([0,1,0])
        self.E3 = np.array([0,0,1])
        # generalized alpha parameters
        self.MAXITERn = 20
        self.MAXITERn_initial = self.MAXITERn   # saving initial value of MAXITERn
        self.r = 0.3
        self.rho_inf = 0.5
        self.rho_infinity_initial = self.rho_inf
        # eq. 72
        self.alpha_m = (2*self.rho_inf-1)/(self.rho_inf+1)
        self.alpha_f = self.rho_inf/(self.rho_inf+1)
        self.gama = 0.5+self.alpha_f-self.alpha_m
        self.beta = 0.25*(0.5+self.gama)**2
        self.tol_n = 1.0e-6     # error tolerance
        # discritization of tau value for finding minimizing one
        self.n_tau = int(1/self.tol_n)
        # mass matrix (constant)
        self.Mdiag = np.array([self.m, self.m, self.m, self.It, self.It, self.Ia])
        self.M = np.diag(self.Mdiag)
        # applied forces (weight)
        self.force = np.array([0, 0, -self.m*self.gr, 0, 0, 0])
        # get the saved lists of unique contacts
        # define the path where the arrays were saved
        unique_contacts_path = os.path.join(os.getcwd(), "unique_contacts")
        # load the arrays
        self.unique_contacts_a = np.load(f'{unique_contacts_path}/unique_contacts_a.npy')
        self.unique_contacts_b = np.load(f'{unique_contacts_path}/unique_contacts_b.npy')
        self.unique_contacts_c = np.load(f'{unique_contacts_path}/unique_contacts_c.npy')
        self.unique_contacts_d = np.load(f'{unique_contacts_path}/unique_contacts_d.npy')
        # save arrays
        self.q_save = np.zeros((1,self.ndof,self.ntime))
        self.u_save = np.zeros((1,self.ndof,self.ntime))
        self.X_save = np.zeros((1,self.nX,self.ntime))
        self.gNdot_save = np.zeros((1,self.nN,self.ntime))
        self.gammaF_save = np.zeros((1,self.nF,self.ntime))
        self.AV_save = np.zeros((1,self.ndof+self.nN+self.nF,self.ntime))
        self.contacts_save = np.zeros((1,5*self.nN,self.ntime))
        # initial position
        # q0 = np.array([self.R_hip-self.R_hoop, 0, 0, 0, 0, 0])
        q0 = np.array([0, 0, self.R_hip-self.R_hoop+0.001, np.pi/2, np.pi/2, 0])
        self.q_save[0,:,0] = q0
        # initial velocity
        # u0 = np.array([-0.1, 0, 0, 0, 0, 10])
        u0 = np.array([1, 0, 0, 0, 0, 0])
        self.u_save[0,:,0] = u0
        # multiple solution parameters
        self.total_leaves = 0
        # array to keep track of bifurcations
        self.bif_tracker = np.empty((0,2))
        
        # creating an output file f to log major happenings
        self.f = open(f"{self.output_path}/log_file.txt",'a')

        # Bind the function to the class
        from contact_constraints import get_contact_constraints
        self.get_contact_constraints = get_contact_constraints.__get__(self)

    def save_arrays(self):
        """Saving arrays."""
        file_name_q = str(f'{self.output_path}/q.mat')
        scipy.io.savemat(file_name_q,dict(q=self.q_save))

        file_name_u = str(f'{self.output_path}/u.mat')
        scipy.io.savemat(file_name_u,dict(u=self.u_save))

        file_name_x_save = str(f'{self.output_path}/x_save.mat')
        scipy.io.savemat(file_name_x_save,dict(X=self.X_save))

        file_name_q_hip = str(f'{self.output_path}/q_hip.mat')
        scipy.io.savemat(file_name_q_hip,dict(q_hip=self.q_hip))

        file_name_contacts = str(f'{self.output_path}/contacts.mat')
        scipy.io.savemat(file_name_contacts,dict(contacts=self.contacts_save))

        file_name_bif_tracker = str(f'{self.output_path}/bif_tracker.mat')
        scipy.io.savemat(file_name_bif_tracker,dict(bif_tracker=self.bif_tracker))


        np.save(f'{self.output_path}/q_save.npy', self.q_save)
        np.save(f'{self.output_path}/u_save.npy', self.u_save)
        np.save(f'{self.output_path}/X_save.npy', self.X_save)
        np.save(f'{self.output_path}/gNdot_save.npy', self.gNdot_save)
        np.save(f'{self.output_path}/gammaF_save.npy', self.gammaF_save)
        np.save(f'{self.output_path}/AV_save.npy', self.AV_save)
        return

    def get_minimizing_tau(self, q_hoop, q_hip):
        """Return the minimizing values of tau describing current or potential contact."""
        # center of hoop
        xbar_hoop = q_hoop[:3]
        # Euler angles of hoop
        psi_hoop = q_hoop[3]
        theta_hoop = q_hoop[4]
        phi_hoop = q_hoop[5]
        # Rotation matrices of hoop
        R1_hoop = np.array([[np.cos(psi_hoop), np.sin(psi_hoop), 0],[-np.sin(psi_hoop), np.cos(psi_hoop), 0],[0, 0, 1]])
        R2_hoop = np.array([[1, 0, 0],[0, np.cos(theta_hoop), np.sin(theta_hoop)],[0, -np.sin(theta_hoop), np.cos(theta_hoop)]])
        R3_hoop = np.array([[np.cos(phi_hoop), np.sin(phi_hoop), 0],[-np.sin(phi_hoop), np.cos(phi_hoop), 0],[0, 0, 1]])
        # {E1, E2, E3} components
        e1_hoop = np.transpose(R3_hoop@R2_hoop@R1_hoop)@self.E1
        e2_hoop = np.transpose(R3_hoop@R2_hoop@R1_hoop)@self.E2

        # center of hip
        xbar_hip = q_hip[:3]
        # Euler angles of hip
        psi_hip = q_hip[3]
        theta_hip = q_hip[4]
        phi_hip = q_hip[5]
        # Rotation matrices of hoop
        R1_hip = np.array([[np.cos(psi_hip), np.sin(psi_hip), 0],[-np.sin(psi_hip), np.cos(psi_hip), 0],[0, 0, 1]])
        R2_hip = np.array([[1, 0, 0],[0, np.cos(theta_hip), np.sin(theta_hip)],[0, -np.sin(theta_hip), np.cos(theta_hip)]])
        R3_hip = np.array([[np.cos(phi_hip), np.sin(phi_hip), 0],[-np.sin(phi_hip), np.cos(phi_hip), 0],[0, 0, 1]])
        # {E1, E2, E3} components
        e3_hip = np.transpose(R3_hip@R2_hip@R1_hip)@self.E3

        # Create an array of possible tau values (step size < algorithm tolerance)
        tau = np.linspace(0, 2*np.pi, num=self.n_tau, endpoint=True)
        # I can find intervals containing the minima and then refine the discretization in these intervals (or use the bisection method)

        # Creating array of hoop points
        # # Reshape tau to (1000000, 1) to enable broadcasting
        u = np.cos(tau)[:, np.newaxis] * e1_hoop + np.sin(tau)[:, np.newaxis] * e2_hoop  # Shape (1000000, 3)
        xM = xbar_hoop+self.R_hoop*u
        # Calculating the value of dh for each point
        dv = np.dot(xM,e3_hip)
        dh_vec = xM-dv[:, np.newaxis]*e3_hip-xbar_hip
        # Compute the norm of each row
        dh = np.linalg.norm(dh_vec, axis=1) # SOMETHING WRONG WITH DH CALCULATION

        # Find the minimizers of dh
        # Find local minima (less than neighbors)
        min_indices = argrelextrema(dh, np.less)[0]
        if min_indices.size == 0:
            # if there is no minimzing index, then either distribution is uniform, or local min happening at edge
            min_indices = [0]
        # Find the minizing value of tau
        minimizing_tau = tau[min_indices]
        # minimizing_dh = dh[min_indices]

        return minimizing_tau

    
    def combine_contact_constraints(self,iter,q_hoop,u_hoop,a_hoop):
        ''' Combine all gap distance, slip speed functions and the gradients and derivatives from both contacts.'''

        # get the minimizing values
        tau = self.get_minimizing_tau(q_hoop, self.q_hip[iter,:])
        
        # Contact constraints and constraint gradients
        # initializing constraints
        gN = np.zeros(self.nN)
        gammaF = np.zeros(self.nF)
        # initialize constraint derivatives
        gNdot = np.zeros(self.nN)
        gNddot = np.zeros(self.nN)
        gammadotF = np.zeros(self.nF)
        # initializing constraint gradients
        WN = np.zeros((self.ndof, self.nN))
        WF = np.zeros((self.ndof, self.nF))

        if np.size(tau) == 2:  # two local minima
            gN[1], gNdot[1], gNddot[1], WN[:,1], gammaF[self.gammaF_lim[1,:]], gammadotF[self.gammaF_lim[1,:]], WF[:,self.gammaF_lim[1,:]] = self.get_contact_constraints(q_hoop,u_hoop,a_hoop,tau[1],self.q_hip[iter,:],self.u_hip[iter,:],self.a_hip[iter,:])
            gN[0], gNdot[0], gNddot[0], WN[:,0], gammaF[self.gammaF_lim[0,:]], gammadotF[self.gammaF_lim[0,:]], WF[:,self.gammaF_lim[0,:]] = self.get_contact_constraints(q_hoop,u_hoop,a_hoop,tau[0],self.a_hip[iter,:],self.u_hip[iter,:],self.a_hip[iter,:])
            # saving values
            # minimizing_tau_save[:,iter] = tau 
            
        elif np.size(tau) == 1:
            # This case is rare if the hoop is not initialized to a horizontal configuration, or the extremum is at the end of the array
            gN[0], gNdot[0], gNddot[0], WN[:,0], gammaF[self.gammaF_lim[0,:]], gammadotF[self.gammaF_lim[0,:]], WF[:,self.gammaF_lim[0,:]] = self.get_contact_constraints(q_hoop,u_hoop,a_hoop,tau[0],self.q_hip[iter,:],self.u_hip[iter,:],self.a_hip[iter,:])
            gN[1] = 1   # >0, no contact, we don't worry about other values
            # saving values
            # minimizing_tau_save[0,iter] = tau.item()
            # CONCERN: nonsmooth jumps in contact functions

        return gN, gNdot, gNddot, WN, gammaF, gammadotF, WF
    
    def get_R(self,iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,*index_sets):
        """Calculates the residual."""

        [prev_a,_,_,_,_,_,_,_,_,_,prev_lambdaN,_,prev_lambdaF] = self.get_X_components(prev_X)
        [a,U,Q,Kappa_g,Lambda_g,lambda_g,Lambda_gamma,lambda_gamma,
            KappaN,LambdaN,lambdaN,LambdaF,lambdaF] = self.get_X_components(X)
        
        # AV - Auxiliary Variables [abar, lambdaNbar, lambdaFbar]
        prev_abar = prev_AV[0:self.ndof]
        prev_lambdaNbar = prev_AV[self.ndof:self.ndof+self.nN]
        prev_lambdaFbar = prev_AV[self.ndof+self.nN:self.ndof+self.nN+self.nF]

        # auxiliary variables update
        # eq. 49
        abar = (self.alpha_f*prev_a+(1-self.alpha_f)*a-self.alpha_m*prev_abar)/(1-self.alpha_m)
        # eq. 96
        lambdaNbar = (self.alpha_f*prev_lambdaN+(1-self.alpha_f)*lambdaN-self.alpha_m*prev_lambdaNbar)/(1-self.alpha_m)
        # eq. 114
        lambdaFbar = (self.alpha_f*prev_lambdaF+(1-self.alpha_f)*lambdaF-self.alpha_m*prev_lambdaFbar)/(1-self.alpha_m)

        AV = np.concatenate((abar,lambdaNbar,lambdaFbar),axis=None)

        # velocity update (73)
        u = prev_u+self.dtime*((1-self.gama)*prev_abar+self.gama*abar)+U
        # position update (73)
        q = prev_q+self.dtime*prev_u+self.dtime**2/2*((1-2*self.beta)*prev_abar+2*self.beta*abar)+Q

        # bilateral constraints at position level
        g = np.zeros((self.ng))
        gdot = np.zeros((self.ng))
        gddot = np.zeros((self.ng))
        Wg = np.zeros((self.ndof,self.ng))

        # bilateral constraints at velocity level
        gamma = np.zeros((self.ngamma))
        gammadot = np.zeros((self.ngamma))
        Wgamma = np.zeros((self.ndof,self.ngamma))

        # normal gap distance constraints and some frictional quantities
        gN, gNdot, gNddot, WN, gammaF, gammaFdot, WF = self.combine_contact_constraints(iter,q,u,a)

        # eq. 44
        ksiN = gNdot+self.eN*prev_gNdot
        # discrete normal percussion eq. 95
        PN = LambdaN+self.dtime*((1-self.gama)*prev_lambdaNbar+self.gama*lambdaNbar)
        # eq. 102
        Kappa_hatN = KappaN+self.dtime**2/2*((1-2*self.beta)*prev_lambdaNbar+2*self.beta*lambdaNbar)

        # eq. 48
        ksiF = gammaF+self.eN*prev_gammaF
        # eq. 113
        PF = LambdaF+self.dtime*((1-self.gama)*prev_lambdaFbar+self.gama*lambdaFbar)    
            
        Rs = np.concatenate(([self.M@a-self.force-Wg@lambda_g-Wgamma@lambda_gamma-WN@lambdaN-WF@lambdaF],
                [self.M@U-Wg@Lambda_g-Wgamma@Lambda_gamma-WN@LambdaN-WF@LambdaF],
                [self.M@Q-Wg@Kappa_g-WN@KappaN-self.dtime/2*(Wgamma@Lambda_gamma+WF@LambdaF)],
                g,
                gdot,
                gddot,
                gamma,
                gammadot),axis=None)
        
        # Contact residual Rc
        R_KappaN = np.zeros(self.nN)   # (129)
        R_LambdaN = np.zeros(self.nN)
        R_lambdaN = np.zeros(self.nN)
        R_LambdaF = np.zeros(self.nF)  # (138)
        R_lambdaF = np.zeros(self.nF)  # (142)

        if index_sets == ():
            A = np.zeros(self.nN, dtype=int)
            B = np.zeros(self.nN, dtype=int)
            C = np.zeros(self.nN, dtype=int)
            D = np.zeros(self.nN, dtype=int)
            E = np.zeros(self.nN, dtype=int)

            for i in range(self.nN):
                # check for contact if blocks are not horizontally detached
                if self.r*gN[i] - Kappa_hatN[i] <=0:
                    A[i] = 1
                    if np.linalg.norm(self.r*ksiF[self.gammaF_lim[i,:]]-PF[self.gammaF_lim[i,:]])<=self.mu_s*(PN[i]):
                        # D-stick
                        D[i] = 1
                        if np.linalg.norm(self.r*gammaFdot[self.gammaF_lim[i,:]]-lambdaF[self.gammaF_lim[i,:]])<=self.mu_s*(lambdaN[i]):
                            # E-stick
                            E[i] = 1
                    if self.r*ksiN[i]-PN[i] <= 0:
                        B[i] = 1
                        if self.r*gNddot[i]-lambdaN[i] <= 0:
                            C[i] = 1
        else:
            A = index_sets[0]
            B = index_sets[1]
            C = index_sets[2]
            D = index_sets[3]
            E = index_sets[4]

        # calculating contact residual
        for k in range(self.nN):
            if A[k]:
                R_KappaN[k] = gN[k]
                if D[k]:
                    R_LambdaF[self.gammaF_lim[k,:]] = ksiF[self.gammaF_lim[k,:]]
                    if E[k]:
                        R_lambdaF[self.gammaF_lim[k,:]] = gammaFdot[self.gammaF_lim[k,:]]
                    else:
                        R_lambdaF[self.gammaF_lim[k,:]] = lambdaF[self.gammaF_lim[k,:]]+self.mu_k*lambdaN[k]*np.sign(gammaFdot[self.gammaF_lim[k,:]])                    
                else:
                    R_LambdaF[self.gammaF_lim[k,:]] = PF[self.gammaF_lim[k,:]]+self.mu_k*PN[k]*np.sign(ksiF[self.gammaF_lim[k,:]])
                    R_lambdaF[self.gammaF_lim[k,:]] = lambdaF[self.gammaF_lim[k,:]]+self.mu_k*lambdaN[k]*np.sign(gammaF[self.gammaF_lim[k,:]])
            else:
                R_KappaN[k] = Kappa_hatN[k]
                R_LambdaF[self.gammaF_lim[k,:]] = PF[self.gammaF_lim[k,:]]
                R_lambdaF[self.gammaF_lim[k,:]] = lambdaF[self.gammaF_lim[k,:]]
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


        Rc = np.concatenate((R_KappaN, R_LambdaN, R_lambdaN, R_LambdaF, R_lambdaF),axis=None)
        
        R = np.concatenate([Rs, Rc],axis=None)


        if index_sets == ():
            # in this case, get_R is called to calculate the actual residual, not as part of calculating the Jacobian
            print(f"A={A}")
            print(f"B={B}")
            print(f"C={C}")
            print(f"D={D}")
            print(f"E={E}")
            return R, AV, q, u, gNdot, gammaF, A, B, C, D, E
        else:
            # in this case, get_R is called as part of calculating the Jacobian for fixed contact regions
            return R, AV, q, u, gNdot, gammaF

    def get_R_J(self,iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,*fixed_contact):
        '''Calculate the Jacobian manually.'''

        epsilon = 1e-6
        fixed_contact = np.squeeze(fixed_contact)

        if fixed_contact.size > 0:
            # here, the contact is fixed if a solve_bifurcation is being run
            contacts_nu = fixed_contact
            A = contacts_nu[0:self.nN]
            B = contacts_nu[self.nN:2*self.nN]
            C = contacts_nu[2*self.nN:3*self.nN]
            D = contacts_nu[3*self.nN:3*self.nN+self.nN]
            E = contacts_nu[3*self.nN+self.nN:3*self.nN+2*self.nN]
            R, AV, q, u, gNdot, gammaF =  self.get_R(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF, A, B, C, D, E)
        else:
            R, AV, q, u, gNdot, gammaF, A, B, C, D, E = self.get_R(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)
            contacts_nu = np.concatenate((A,B,C,D,E),axis=None)

        # Initializing the Jacobian
        J = np.zeros((self.nX,self.nX))
        I = np.identity(self.nX)

        # Constructing the Jacobian column by column
        for i in range(self.nX):
            # print(i)
            R_plus_epsilon,_,_,_,_,_ = self.get_R(iter,X+epsilon*I[:,i],prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF, A, B, C, D, E)
            J[:,i] = (R_plus_epsilon-R)/epsilon

        return R, AV, q, u, gNdot, gammaF, J, contacts_nu

    def update(self,leaf,iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,*fixed_contact):
        """Takes components at time t and return values at time t+dt"""

        nu = 0
        print(f"Update is called for iter = {iter} and leaf = {leaf}")
        self.f.write(f"Update is called for iter = {iter} and leaf = {leaf}")
        
        X = prev_X
        R, AV, q, u, gNdot, gammaF, J, contacts_nu = self.get_R_J(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,*fixed_contact)

        contacts = np.zeros((self.MAXITERn+1,3*self.nN+2*self.nN),dtype=int)
        contacts[nu,:] = contacts_nu
        self.contacts_save[leaf,:,iter] = contacts_nu

        norm_R = np.linalg.norm(R,np.inf)
        print(f"norm(R) = {norm_R}")

        try:

            while np.abs(np.linalg.norm(R,np.inf))>self.tol_n and nu<self.MAXITERn:
                # Newton Update
                X = X-np.linalg.solve(J,R)
                # Calculate new EOM and residual
                nu = nu+1

                R, AV, q, u, gNdot, gammaF, J, contacts_nu = self.get_R_J(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,fixed_contact)
                
                contacts[nu,:] = contacts_nu
                self.contacts_save[leaf,:,iter] = contacts_nu
                    
                norm_R = np.linalg.norm(R,np.inf)
                print(f"nu = {nu}")
                print(f"norm(R) = {norm_R}")

                if norm_R>10**6:
                    # the Jacobian is blowing up
                    # (I am assuming this is happening because contact region is fixed, 
                    raise JacobianBlowingUpError
                
            if nu == self.MAXITERn:
                self.f.write(f"  Raising MaxNewtonIterAttainedError")
                raise MaxNewtonIterAttainedError
        
            return X,AV,q,u,gNdot,gammaF
        
        except (JacobianBlowingUpError,MaxNewtonIterAttainedError) as e:
            if fixed_contact == ():
                unique_contacts = np.unique(contacts, axis=0)
                # because if the number of contact regions is 6 which is the original number
                do_not_unpack = True
                return unique_contacts, do_not_unpack
            else:
                raise e
        
        except np.linalg.LinAlgError as e:
            # maybe update rho_inf here
            self.f.write(f"  Raising np.linalg.LinAlgError")
            raise e
        
        except Exception as e:
            print(e)
            self.f.write(f"  Raising exception {e}")
            raise e
        
    def update_rho_inf(self):
        '''Update the numerical parameter rho_inf.'''
        self.rho_inf = self.rho_inf+0.05  #0.01
        print(self.rho_inf)
        self.f.write(f"  Updating rho_inf to {self.rho_inf}")
        if np.abs(self.rho_inf - self.rho_infinity_initial) < 0.001:
            print("possibility of infinite loop")
            self.f.write(f"  Raising RhoInfInfiniteLoop error")
            raise RhoInfInfiniteLoop
        if self.rho_inf > 1.001:
            self.rho_inf = 0
        # eq. 72
        self.alpha_m = (2*self.rho_inf-1)/(self.rho_inf+1)
        self.alpha_f = self.rho_inf/(self.rho_inf+1)
        self.gama = 0.5+self.alpha_f-self.alpha_m
        self.beta = 0.25*(0.5+self.gama)**2

    def get_X_components(self,X):
        '''Getting the components of the array X.'''
        a = X[0:self.ndof]
        U = X[self.ndof:2*self.ndof]
        Q = X[2*self.ndof:3*self.ndof]
        Kappa_g = X[3*self.ndof:3*self.ndof+self.ng]
        Lambda_g = X[3*self.ndof+self.ng:3*self.ndof+2*self.ng]
        lambda_g = X[3*self.ndof+2*self.ng:3*self.ndof+3*self.ng]
        Lambda_gamma = X[3*self.ndof+3*self.ng:3*self.ndof+3*self.ng+self.ngamma]
        lambda_gamma = X[3*self.ndof+3*self.ng+self.ngamma:3*self.ndof+3*self.ng+2*self.ngamma]
        Kappa_N = X[3*self.ndof+3*self.ng+2*self.ngamma:3*self.ndof+3*self.ng+2*self.ngamma+self.nN]
        Lambda_N = X[3*self.ndof+3*self.ng+2*self.ngamma+self.nN:3*self.ndof+3*self.ng+2*self.ngamma+2*self.nN]
        lambda_N = X[3*self.ndof+3*self.ng+2*self.ngamma+2*self.nN:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN]
        Lambda_F = X[3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+self.nF]
        lambda_F = X[3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+self.nF:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+2*self.nF]
        return a,U,Q,Kappa_g,Lambda_g,lambda_g,Lambda_gamma,lambda_gamma,\
            Kappa_N,Lambda_N,lambda_N,Lambda_F,lambda_F

    def increment_saved_arrays(self,leaf):
        '''Increment saved arrays due to a bifurcation.'''

        self.f.write(f"  Saved arrays are incremented at leaf={leaf}.")
        
        self.save_arrays()

        # increment saved arrays
        
        q_save_addition = np.tile(self.q_save[leaf:leaf+1, :, :], (1, 1, 1))  # shape: (1, :, :)
        self.q_save = np.insert(self.q_save, leaf + 1, q_save_addition, axis=0)
        # q_save_addition = np.tile(self.q_save[leaf,:,:],(1,1,1))
        # self.q_save = np.vstack((self.q_save,q_save_addition))

        u_save_addition = np.tile(self.u_save[leaf:leaf+1, :, :], (1, 1, 1))  # shape: (1, :, :)
        self.u_save = np.insert(self.u_save, leaf + 1, u_save_addition, axis=0)
        # u_save_addition = np.tile(self.u_save[leaf,:,:],(1,1,1))
        # self.u_save = np.vstack((self.u_save,u_save_addition))

        X_save_addition = np.tile(self.X_save[leaf:leaf+1, :, :], (1, 1, 1))  # shape: (1, :, :)
        self.X_save = np.insert(self.X_save, leaf + 1, X_save_addition, axis=0)
        # X_save_addition = np.tile(self.X_save[leaf,:,:],(1,1,1))
        # self.X_save = np.vstack((self.X_save,X_save_addition))

        gNdot_save_addition = np.tile(self.gNdot_save[leaf:leaf+1, :, :], (1, 1, 1))  # shape: (1, :, :)
        self.gNdot_save = np.insert(self.gNdot_save, leaf + 1, gNdot_save_addition, axis=0)
        # gNdot_save_addition = np.tile(self.gNdot_save[leaf,:,:],(1,1,1))
        # self.gNdot_save = np.vstack((self.gNdot_save,gNdot_save_addition))

        gammaF_save_addition = np.tile(self.gammaF_save[leaf:leaf+1, :, :], (1, 1, 1))  # shape: (1, :, :)
        self.gammaF_save = np.insert(self.gammaF_save, leaf + 1, gammaF_save_addition, axis=0)
        # gammaF_save_addition = np.tile(self.gammaF_save[leaf,:,:],(1,1,1))
        # self.gammaF_save = np.vstack((self.gammaF_save,gammaF_save_addition))

        AV_save_addition = np.tile(self.AV_save[leaf:leaf+1, :, :], (1, 1, 1))  # shape: (1, :, :)
        self.AV_save = np.insert(self.AV_save, leaf + 1, AV_save_addition, axis=0)
        # AV_save_addition = np.tile(self.AV_save[leaf,:,:],(1,1,1))
        # self.AV_save = np.vstack((self.AV_save,AV_save_addition))

        contacts_save_addition = np.tile(self.contacts_save[leaf:leaf+1, :, :], (1, 1, 1))  # shape: (1, :, :)
        self.contacts_save = np.insert(self.contacts_save, leaf + 1, contacts_save_addition, axis=0)
        # contacts_save_addition = np.tile(self.contacts_save[leaf,:,:],(1,1,1))
        # self.contacts_save = np.vstack((self.contacts_save,contacts_save_addition))
    
    def solve_open_contact(self, iter, leaf):
        '''checking for no contact'''

        self.f.write(f"  Checking for open contact.")

        prev_X = self.X_save[leaf,:,iter-1]
        prev_AV = self.AV_save[leaf,:,iter-1]
        prev_q = self.q_save[leaf,:,iter-1]
        prev_u = self.u_save[leaf,:,iter-1]
        prev_gNdot = self.gNdot_save[leaf,:,iter-1]
        prev_gammaF = self.gammaF_save[leaf,:,iter-1]

        open_contact = np.zeros(10)

        try:
            X,AV,q,u,gNdot,gammaF = self.update(leaf,iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,open_contact)
        except (np.linalg.LinAlgError, JacobianBlowingUpError, MaxNewtonIterAttainedError) as e:
            print(e)
            self.f.write(f"  Error {e} raised. The contact is not open.")
            raise NoOpenContactError
            
        # calculate residual with these values
        R, _, _, _, _, _, A, B, C, D, E = self.get_R(iter,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)
        output_contacts = np.concatenate((A,B,C,D,E),axis=None)
        if np.abs(np.linalg.norm(R,np.inf))<self.tol_n and (output_contacts==open_contact).all():
            self.q_save[leaf,:,iter] = q
            self.u_save[leaf,:,iter] = u
            self.X_save[leaf,:,iter] = X
            self.gNdot_save[leaf,:,iter] = gNdot
            self.gammaF_save[leaf,:,iter] = gammaF
            self.AV_save[leaf,:,iter] = AV
            self.save_arrays()

            print(f'The contacts are all open.')
            return
        else:
            self.f.write(f"  Raised NoOpenContactError. The contact is not open.")
            raise NoOpenContactError
        
        
    def solve_fixed_contacts(self, iter, leaf, unique_contacts):
        '''Solving for a set of fixed contact regions.'''

        prev_X = self.X_save[leaf,:,iter-1]
        prev_AV = self.AV_save[leaf,:,iter-1]
        prev_q = self.q_save[leaf,:,iter-1]
        prev_u = self.u_save[leaf,:,iter-1]
        prev_gNdot = self.gNdot_save[leaf,:,iter-1]
        prev_gammaF = self.gammaF_save[leaf,:,iter-1]

        # remove rows that are all zeros (in case there are any)
        unique_contacts = unique_contacts[~np.all(unique_contacts == 0, axis=1)]
        n_unique_contacts = np.shape(unique_contacts)[0]

        self.f.write(f"  Solving for fixed contacts at iter = {iter}, leaf = {leaf} and n_unique_contacts = {n_unique_contacts}.")
                    
        convergence_counter = 0
        nonconvergence_counter = 0
        
        for i in range(n_unique_contacts):
            contact = unique_contacts[i,:]

            try:
                X,AV,q,u,gNdot,gammaF = self.update(leaf,iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,contact)

                if convergence_counter == 0:
                    self.f.write(f"  Convergence! This is the first converged leaf. Do not increment saved arrays.")
                    print(f'This is the first converged leaf. Do not increment saved arrays.')
                else:
                    self.f.write(f"  Convergence! Increment saved arrays.")
                    print(f'Increment saved arrays.')
                    self.increment_saved_arrays(leaf)
                    # increment at end of saved arrays
                    self.total_leaves += 1
                    leaf += 1
                
                self.q_save[leaf,:,iter] = q
                self.u_save[leaf,:,iter] = u
                self.X_save[leaf,:,iter] = X
                self.gNdot_save[leaf,:,iter] = gNdot
                self.gammaF_save[leaf,:,iter] = gammaF
                self.AV_save[leaf,:,iter] = AV

                convergence_counter += 1

                self.save_arrays()
                print(f'Success.')

            except (np.linalg.LinAlgError, JacobianBlowingUpError, MaxNewtonIterAttainedError) as e:
                self.f.write(f"  This leaf did not converge.  The error {e} was raised.")
                nonconvergence_counter += 1

        if nonconvergence_counter == n_unique_contacts:
            self.f.write(f"  None of the leaves conveged. Raised NoBifurcationConvergence error.")
            print(f"Solution 1 did not work. None of the leaves converged.")
            raise NoBifurcationConvergence
        else:
            return convergence_counter

    def time_update(self, iter, leaf):

        self.f.write(f"Running time update at iter = {iter}, leaf = {leaf}.")

        prev_X = self.X_save[leaf,:,iter-1]
        prev_AV = self.AV_save[leaf,:,iter-1]
        prev_q = self.q_save[leaf,:,iter-1]
        prev_u = self.u_save[leaf,:,iter-1]
        prev_gNdot = self.gNdot_save[leaf,:,iter-1]
        prev_gammaF = self.gammaF_save[leaf,:,iter-1]

        try:
            X,AV,q,u,gNdot,gammaF = self.update(leaf,iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)

            self.q_save[leaf,:,iter] = q
            self.u_save[leaf,:,iter] = u
            self.X_save[leaf,:,iter] = X
            self.gNdot_save[leaf,:,iter] = gNdot
            self.gammaF_save[leaf,:,iter] = gammaF
            self.AV_save[leaf,:,iter] = AV
            self.save_arrays()

            self.f.write(f'Success. No issues. leaf = {leaf}. iter = {iter} converged.')
            print(f'Success. No issues. leaf = {leaf}. iter = {iter}.')

            convergence_counter = 1
            return convergence_counter
        
        except np.linalg.LinAlgError as e:
            self.f.write(f'Raised np.linalg.LinAlgError')
            print(e)

            try:
                # solution 2: looping over all possible contact configs
                self.f.write(f'  Looping over all possible contact configs.')
                unique_contacts = np.empty((0, 10))
                # unique_contacts = np.vstack([unique_contacts,self.unique_contacts_a])
                unique_contacts = np.vstack([unique_contacts,self.unique_contacts_b])
                unique_contacts = np.vstack([unique_contacts,self.unique_contacts_c])
                unique_contacts = np.vstack([unique_contacts,self.unique_contacts_d])

                convergence_counter = self.solve_fixed_contacts(iter,leaf,unique_contacts)

                self.f.write(f'  Success. Looped over all possible contact configs. leaf = {leaf}. iter = {iter}. convergence_counter = {convergence_counter}.')
                print(f'Success. Looped over all possible contact configs. leaf = {leaf}. iter = {iter}. convergence_counter = {convergence_counter}.')

                return convergence_counter
        
            except Exception as e:
                self.f.write(f'  Raised error {e}. Deleting leaf = {leaf}.')
                print(e)
                self.delete_leaf(leaf)

                print(f"I need to implement increasing maxiter_n or increasing rho_inf.")
                # solution: increment maxitern
                # solution: increment rho_inf
                # self.update_rho_inf()

                convergence_counter = 0
                return convergence_counter

        except ValueError as e:

            unique_contacts,_ = self.update(leaf,iter,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)

            try:
                # solution 0: checking for no contact
                self.solve_open_contact(iter,leaf)

                self.f.write(f'  Success. Open contact. leaf = {leaf}. iter = {iter}.')
                print(f'Success. Open contact. leaf = {leaf}. iter = {iter}.')

                convergence_counter = 1
                return convergence_counter

            except Exception as e:
                print(e)
                print(f"Solution 0 did not work. Contact is not open.")
                print(f"Loop over attained contact configurations with closed contact.")
                self.f.write(f'  Contact is not open. Loop over attained contact configurations with closed contact.')

                try:
                    # solution 1: looping over attained contact configs
                    convergence_counter = self.solve_fixed_contacts(iter,leaf,unique_contacts)

                    self.f.write(f'  Success. Looped over attained contact configs. leaf = {leaf}. iter = {iter}. convergence_counter = {convergence_counter}.')
                    print(f'Success. Looped over attained contact configs. leaf = {leaf}. iter = {iter}. convergence_counter = {convergence_counter}.')

                    return convergence_counter
                    
                except Exception as e:
                    print(e)
                    print(f"Solution 1 did not work. None of the attained contact regions converged.")
                    print(f"Loop over attained contact configurations with closed contact.")
                    self.f.write(f"  None of the attained contact regions converged. Loop over attained contact configurations with closed contact.")

                    try:
                        # solution 2: looping over all possible contact configs
                        unique_contacts = np.empty((0, 10))
                        # unique_contacts = np.vstack([unique_contacts,self.unique_contacts_a])
                        unique_contacts = np.vstack([unique_contacts,self.unique_contacts_b])
                        unique_contacts = np.vstack([unique_contacts,self.unique_contacts_c])
                        unique_contacts = np.vstack([unique_contacts,self.unique_contacts_d])

                        convergence_counter = self.solve_fixed_contacts(iter,leaf,unique_contacts)

                        self.f.write(f"  Success. Looped over all possible contact configs. leaf = {leaf}. iter = {iter}. convergence_counter = {convergence_counter}.")
                        print(f'Success. Looped over all possible contact configs. leaf = {leaf}. iter = {iter}. convergence_counter = {convergence_counter}.')

                        return convergence_counter

                    except Exception as e:
                        print(e)
                        self.delete_leaf(leaf)
                        self.f.write(f"  Raised error {e}. Deleted leaf = {leaf}.")

                        print(f"I need to implement increasing maxiter_n or increasing rho_inf.")
                        # solution: increment maxitern
                        # solution: increment rho_inf
                        # self.update_rho_inf()

                        convergence_counter = 0
                        return convergence_counter
                    
    def delete_leaf(self,leaf):
        ''' delete leaf that did not converge: decrement saved arrays'''
        self.q_save = np.delete(self.q_save,leaf,0)
        self.u_save = np.delete(self.u_save,leaf,0)
        self.X_save = np.delete(self.X_save,leaf,0)
        self.gNdot_save = np.delete(self.gNdot_save,leaf,0)
        self.gammaF_save = np.delete(self.gammaF_save,leaf,0)
        self.AV_save = np.delete(self.AV_save,leaf,0)
        self.contacts_save = np.delete(self.contacts_save,leaf,0)
        self.total_leaves = self.total_leaves-1
                            
    def solve_A(self):
        leaf = 0

        while leaf <= self.total_leaves:
            iter = 1
            self.f.write(f"  Increment leaf = {leaf}. iter = {iter}.")
            while iter < self.ntime:
                convergence_counter = self.time_update(iter, leaf)
                self.bif_tracker = np.vstack([leaf,iter,convergence_counter])
                iter += 1
            leaf += convergence_counter

    def solve_B(self):
        iter = 1

        while iter <= self.ntime:
            leaf = 0
            self.f.write(f"  Increment iter = {iter}. leaf = {leaf}.")
            while leaf < self.total_leaves:
                convergence_counter = self.time_update(iter, leaf)
                self.bif_tracker = np.vstack([leaf,iter,convergence_counter])
                leaf += convergence_counter
            iter += 1

# # update initial value
# self.rho_infinity_initial = self.rho_inf 
# # reset initial value
# self.MAXITERn = self.MAXITERn_initial
# # reset n_tau value
# self.n_tau = int(1/self.tol_n)
        

# hoop sticking and rotating, mu_s=10**9, u0 = np.array([-0.1, 0, 0, 0, 0, 10])
# # Test ibi and bbb
test = Simulation(ntime = 300, mu_s=10**9, mu_k=0.3, eN=0, eF=0, max_leaves=5)
test.solve_A()