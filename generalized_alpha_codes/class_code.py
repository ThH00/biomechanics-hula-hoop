# I will remove time tracking for now.


import numpy as np
import time
import os
import argparse
from scipy.signal import argrelextrema
import scipy.io

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
    """This exception is raised when the maximum number of run hours specified by the user is exceeded."""
    def __init__(self, message="This exception is raised when the maximum run time is exceeded."):
        self.message = message
        super().__init__(self.message)

class MaxLeavesAttained(Exception):
    """This exception is raised when the maximum number of leaves specified by the user is exceeded."""
    def __init__(self, message="This exception is raised when the maximum number of leaves is exceeded."):
        self.message = message
        super().__init__(self.message)

class NoLocalMinima(Exception):
    def __init__(self, message="The distance between the hoop and the hip has less then 1 or more than 2 local minima."):
        super().__init__(message)

class Simulation:
    def __init__(self, ntime = 5, mu_s=1, mu_k=0.3, eN=0, eF=0, max_leaves=5):
        # path for outputs
        self.output_path = os.path.join(os.getcwd(), "outputs/multiple_solutions")  # Output path
        os.makedirs(self.output_path, exist_ok=True)
        # friction coefficients
        self.mu_s = mu_s    # Static friction coefficient
        self.mu_k = mu_k    # Kinetic friction coefficient
        # restitution coefficients
        self.eN = eN        # normal coefficient of restitution
        self.eF = eF        # friction coefficient of retitution
        # multiple solution parameters
        self.max_leaves = max_leaves
        self.bif_counter = 0
        self.leaves_counter = 0
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
        self.xbar_hip = np.zeros((self.ntime,3))
        self.vbar_hip = np.zeros((self.ntime,3))
        self.abar_hip = np.zeros((self.ntime,3))
        self.omega_hip = np.array([0,0,0])   # angular velocity of hip
        self.alpha_hip = np.array([0,0,0])   # angular acceleration of hip
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
        # initial position
        q0 = np.array([self.R_hip-self.R_hoop, 0, 0, 0, 0, 0])
        self.q_save[0,:,0] = q0
        # initial velocity
        u0 = np.array([-0.1, 0, 0, 0, 0, 10])
        self.u_save[0,:,0] = u0


    def save_arrays(self):
        """Saving arrays."""
        file_name_q = str(f'{self.output_path}/q.mat')
        scipy.io.savemat(file_name_q,dict(q=self.q_save))

        file_name_u = str(f'{self.output_path}/u.mat')
        scipy.io.savemat(file_name_u,dict(u=self.u_save))

        file_name_x_save = str(f'{self.output_path}/x_save.mat')
        scipy.io.savemat(file_name_x_save,dict(X=self.X_save))

        file_name_xbar_hip = str(f'{self.output_path}/xbar_hip.mat')
        scipy.io.savemat(file_name_xbar_hip,dict(xbar_hip=self.xbar_hip))


        np.save(f'{self.output_path}/q_save.npy', self.q_save)
        np.save(f'{self.output_path}/u_save.npy', self.u_save)
        np.save(f'{self.output_path}/X_save.npy', self.X_save)
        np.save(f'{self.output_path}/gNdot_save.npy', self.gNdot_save)
        np.save(f'{self.output_path}/gammaF_save.npy', self.gammaF_save)
        np.save(f'{self.output_path}/AV_save.npy', self.AV_save)
        return

    def get_minimizing_tau(self, q, xbar_hip):
        """Return the minimizing values of tau describing current or potential contact."""
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
        e1 = np.transpose(R3@R2@R1)@self.E1
        e2 = np.transpose(R3@R2@R1)@self.E2

        # Create an array of possible tau values (step size < algorithm tolerance)
        tau = np.linspace(0, 2*np.pi, num=self.n_tau, endpoint=True)
        # I can find intervals containing the minima and then refine the discretization in these intervals (or use the bisection method)

        # Creating array of hoop points
        # # Reshape tau to (1000000, 1) to enable broadcasting
        u = np.cos(tau)[:, np.newaxis] * e1 + np.sin(tau)[:, np.newaxis] * e2  # Shape (1000000, 3)

        xM = xbar_hoop+self.R_hoop*u

        # Calculating the value of dH for each point
        dv = np.dot(xM,self.E3)
        temp = xM-dv[:, np.newaxis]*self.E3-xbar_hip
        # Compute the norm of each row
        dh = np.linalg.norm(temp, axis=1)

        # Find the minimizers of dh
        # Find local minima (less than neighbors)
        min_indices = argrelextrema(dh, np.less)[0]
        # Find the minizing value of tau
        minimizing_tau = tau[min_indices]

        return minimizing_tau
    
    def get_contact_constraints(self, q,u,a,tau,xbar_hip,vbar_hip,abar_hip):
        '''Get gap distance, slip speed functions and their gradients and derivatives at each contact'''
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
        
        WN = np.zeros(self.ndof)
        WF = np.zeros((self.ndof,2))

        # Rotation matrices
        R1 = np.array([[np.cos(psi), np.sin(psi), 0],[-np.sin(psi), np.cos(psi), 0],[0, 0, 1]])
        R2 = np.array([[1, 0, 0],[0, np.cos(theta), np.sin(theta)],[0, -np.sin(theta), np.cos(theta)]])
        R3 = np.array([[np.cos(phi), np.sin(phi), 0],[-np.sin(phi), np.cos(phi), 0],[0, 0, 1]])
        # {E1, E2, E3} components
        e1p = np.transpose(R1)@self.E1
        e1 = np.transpose(R3@R2@R1)@self.E1
        e2 = np.transpose(R3@R2@R1)@self.E2
        e3 = np.transpose(R3@R2@R1)@self.E3

        omega_hoop = psidot*self.E3+thetadot*e1p+phidot*e3

        e1pdot = np.cross(psidot*self.E3,e1p)
        e3dot = np.cross(omega_hoop,e3)

        alpha_hoop = psiddot*self.E3+thetaddot*e1p+phiddot*e3+thetadot*e1pdot+phidot*e3dot

        tau_dot = 0
        tau_ddot = 0

        u_corrotational = tau_dot*(-np.sin(tau)*e1+np.cos(tau)*e2)
        u_double_corrotational = tau_ddot*(-np.sin(tau)*e1+np.cos(tau)*e2)

        u = np.cos(tau)*e1+np.sin(tau)*e2
        udot = u_corrotational+np.cross(omega_hoop,u)
        uddot = u_double_corrotational+2*np.cross(omega_hoop,u_corrotational)+np.cross(omega_hoop,np.cross(omega_hoop,u))+np.cross(alpha_hoop,u)

        xM = xbar_hoop+self.R_hoop*u
        vM = vbar_hoop+self.R_hoop*udot
        aM = abar_hoop+self.R_hoop*uddot

        # vertical components of vector from hip center to point on hoop
        H = xM-xbar_hip-np.dot(xM-xbar_hip,self.E3)*self.E3
        H_dot = vM-vbar_hip-np.dot(vM-vbar_hip,self.E3)*self.E3
        H_ddot = aM-abar_hip-np.dot(aM-abar_hip,self.E3)*self.E3

        norm_H = np.linalg.norm(H)
        gN = norm_H-self.R_hip
        gNdot = np.dot(H,H_dot)/norm_H
        gNddot = (np.dot(H_dot,H_dot)+np.dot(H,H_ddot))/norm_H

        WN[0] = 2*np.dot(H,self.E1)
        WN[1] = 2*np.dot(H,self.E2)
        WN[2] = 0

        de1_dpsi = np.cos(phi)*(self.E2*np.cos(psi) - self.E1*np.sin(psi)) - np.cos(theta)*np.sin(phi)*(self.E1*np.cos(psi) + self.E2*np.sin(psi))
        de2_dpsi = - np.sin(phi)*(self.E2*np.cos(psi) - self.E1*np.sin(psi)) - np.cos(phi)*np.cos(theta)*(self.E1*np.cos(psi) + self.E2*np.sin(psi))
        de1_dtheta = np.sin(phi)*(self.E3*np.cos(theta) - np.sin(theta)*(self.E2*np.cos(psi) - self.E1*np.sin(psi)))
        de2_dtheta = np.cos(phi)*(self.E3*np.cos(theta) - np.sin(theta)*(self.E2*np.cos(psi) - self.E1*np.sin(psi)))
        de1_dphi = np.cos(phi)*(self.E3*np.sin(theta) + np.cos(theta)*(self.E2*np.cos(psi) - self.E1*np.sin(psi))) - np.sin(phi)*(self.E1*np.cos(psi) + self.E2*np.sin(psi))
        de2_dphi = - np.cos(phi)*(self.E1*np.cos(psi) + self.E2*np.sin(psi)) - np.sin(phi)*(self.E3*np.sin(theta) + np.cos(theta)*(self.E2*np.cos(psi) - self.E1*np.sin(psi)))

        dxM_dpsi = np.cos(tau)*de1_dpsi+np.sin(tau)*de2_dpsi
        dxM_dtheta = np.cos(tau)*de1_dtheta+np.sin(tau)*de2_dtheta
        dxM_dphi = np.cos(tau)*de1_dphi+np.sin(tau)*de2_dphi

        WN[3] = 2*np.dot(H,dxM_dpsi-np.dot(dxM_dpsi,self.E3)*self.E3)
        WN[4] = 2*np.dot(H,dxM_dtheta-np.dot(dxM_dtheta,self.E3)*self.E3)
        WN[5] = 2*np.dot(H,dxM_dphi-np.dot(dxM_dphi,self.E3)*self.E3)

        gammaF1 = -self.R_hips*self.omega_hip[0]*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.R_hips*self.omega_hip[1]*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(phidot*np.sin(psi)*np.sin(theta) + thetadot*np.cos(psi)) - self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(-phidot*np.sin(theta)*np.cos(psi) + thetadot*np.sin(psi)) - vbar_hip[2] + vbar_hoop[2]
        gammaF2 = (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*(self.R_hips*self.omega_hip[2]*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - self.R_hoop*(phidot*np.cos(theta) + psidot)*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) + self.R_hoop*(-phidot*np.sin(theta)*np.cos(psi) + thetadot*np.sin(psi))*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) - self.omega_hip[1]*(-self.R_hips*xbar_hip[2]/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) + xbar_hoop[2]) - vbar_hip[0] + vbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*(-self.R_hips*self.omega_hip[2]*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.R_hoop*(phidot*np.cos(theta) + psidot)*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - self.R_hoop*(phidot*np.sin(psi)*np.sin(theta) + thetadot*np.cos(psi))*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) + self.omega_hip[0]*(-self.R_hips*xbar_hip[2]/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) + xbar_hoop[2]) - vbar_hip[1] + vbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
        gammadotF1 = -self.R_hips*self.alpha_hip[0]*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.R_hips*self.alpha_hip[1]*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - abar_hip[2] + abar_hoop[2] + phiddot*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.sin(psi)*np.sin(theta) + self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(theta)*np.cos(psi)) + thetaddot*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(psi) - self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(psi))
        gammadotF2 = -abar_hip[0]*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hip[1]*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + abar_hoop[0]*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - abar_hoop[1]*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - self.alpha_hip[0]*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*(-self.R_hips*xbar_hip[2]/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) + xbar_hoop[2])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.alpha_hip[1]*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*(self.R_hips*xbar_hip[2]/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi)) - xbar_hoop[2])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.alpha_hip[2]*(self.R_hips*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**1.0 + self.R_hips*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**1.0) + phiddot*((-self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(theta) - self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(theta)*np.cos(psi))*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.cos(theta) - self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(psi)*np.sin(theta))*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5) + psiddot*(-self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5) + thetaddot*(self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*np.sin(psi)/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*np.cos(psi)/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5)
        
        WF[0,0] = 0
        WF[0,1] = 0
        WF[0,2] = 1
        WF[0,3] = 0
        WF[0,4] = self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(psi) - self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(psi)
        WF[0,5] = self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.sin(psi)*np.sin(theta) + self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.sin(theta)*np.cos(psi)

        WF[1,0] = (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
        WF[1,1] = -(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
        WF[1,2] = 0
        WF[1,3] = -self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
        WF[1,4] = self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])*np.sin(psi)/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 + self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])*np.cos(psi)/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5
        WF[1,5] = (-self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau))*np.cos(theta) - self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(theta)*np.cos(psi))*(self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5 - (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau))*np.cos(theta) - self.R_hoop*(np.sin(phi)*np.sin(theta)*np.cos(tau) + np.sin(tau)*np.sin(theta)*np.cos(phi))*np.sin(psi)*np.sin(theta))*(self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])/(xbar_hip[2]**2 + (self.R_hoop*((-np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.cos(theta))*np.sin(tau) + (np.sin(phi)*np.cos(psi)*np.cos(theta) + np.sin(psi)*np.cos(phi))*np.cos(tau)) - xbar_hip[1] + xbar_hoop[1])**2 + (self.R_hoop*((-np.sin(phi)*np.cos(psi) - np.sin(psi)*np.cos(phi)*np.cos(theta))*np.sin(tau) + (-np.sin(phi)*np.sin(psi)*np.cos(theta) + np.cos(phi)*np.cos(psi))*np.cos(tau)) - xbar_hip[0] + xbar_hoop[0])**2)**0.5

        gammaF = np.array([gammaF1, gammaF2])
        gammadotF = np.array([gammadotF1, gammadotF2])
        
        return gN, gNdot, gNddot, WN, gammaF, gammadotF, WF
    
    def combine_contact_constraints(self,q,u,a):
        ''' Combine all gap distance, slip speed functions and the gradients and derivatives from both contacts.'''

        # get the minimizing values
        tau = self.get_minimizing_tau(q,self.xbar_hip[iter,:])
        
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
            gN[0], gNdot[0], gNddot[0], WN[:,0], gammaF[self.gammaF_lim[0,:]], gammadotF[self.gammaF_lim[0,:]], WF[:,self.gammaF_lim[0,:]] = self.get_contact_constraints(q,u,a,tau[0],self.xbar_hip[iter,:],self.vbar_hip[iter,:],self.abar_hip[iter,:])
            gN[1], gNdot[1], gNddot[1], WN[:,1], gammaF[self.gammaF_lim[1,:]], gammadotF[self.gammaF_lim[1,:]], WF[:,self.gammaF_lim[1,:]] = self.get_contact_constraints(q,u,a,tau[1],self.xbar_hip[iter,:],self.vbar_hip[iter,:],self.abar_hip[iter,:])
            # saving values
            # minimizing_tau_save[:,iter] = tau 
            
        elif np.size(tau) == 1:
            # This case is rare if the hoop is not initialized to a horizontal configuration
            gN[0], gNdot[0], gNddot[0], WN[:,0], gammaF[self.gammaF_lim[0,:]], gammadotF[self.gammaF_lim[0,:]], WF[:,self.gammaF_lim[0,:]] = self.get_contact_constraints(q,u,a,tau[0],self.xbar_hip[iter,:],self.vbar_hip[iter,:],self.abar_hip[iter,:])
            gN[1] = 1   # >0, no contact, we don't worry about other values
            # saving values
            # minimizing_tau_save[0,iter] = tau.item()
            # CONCERN: nonsmooth jumps in contact functions
        else:
            # raise error
            # this error might be raised when the hoop is horizontal and centered at the hip, in which case there is no contact between hoop and hip and code proceeds normally
            raise NoLocalMinima()

        return gN, gNdot, gNddot, WN, gammaF, gammadotF, WF
    
    def get_R(self,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,*index_sets):
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
        gN, gNdot, gNddot, WN, gammaF, gammaFdot, WF = self.combine_contact_constraints(q,u,a)

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

    def get_R_J(self,X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,*fixed_contact):
        '''Calculate the Jacobian manually.'''

        epsilon = 1e-6
        fixed_contact_regions = False

        if fixed_contact != ():
            # here, the contact is fixed if a solve_bifurcation is being run
            fixed_contact = fixed_contact[0]
            fixed_contact_regions = True
            A = fixed_contact[0:self.nN]
            B = fixed_contact[self.nN:2*self.nN]
            C = fixed_contact[2*self.nN:3*self.nN]
            D = fixed_contact[3*self.nN:3*self.nN+self.nN]
            E = fixed_contact[3*self.nN+self.nN:3*self.nN+2*self.nN]
            R, AV, q, u, gNdot, gammaF =  self.get_R(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF, A, B, C, D, E)
        else:
            R, AV, q, u, gNdot, gammaF, A, B, C, D, E = self.get_R(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)
            contacts_nu = np.concatenate((A,B,C,D,E),axis=None)

        # Initializing the Jacobian
        J = np.zeros((self.nX,self.nX))
        I = np.identity(self.nX)

        # Constructing the Jacobian column by column
        for i in range(self.nX):
            # print(i)
            R_plus_epsilon,_,_,_,_,_ = self.get_R(X+epsilon*I[:,i],prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF, A, B, C, D, E)
            J[:,i] = (R_plus_epsilon-R)/epsilon

        if fixed_contact_regions:
            return R, AV, q, u, gNdot, gammaF, J
        else:
            # return the contact regions 'contacts_nu' to be saved in case they are needed (in the case of unconverged iterations)
            return R, AV, q, u, gNdot, gammaF, J, contacts_nu

    def update(self,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,*fixed_contact):
        """Takes components at time t and return values at time t+dt"""
        
        nu = 0
        X = prev_X
        
        if fixed_contact != ():
            # the contact region is fixed if solve_bifuration is calling update 
            # the fixed_contact data is inputted into get_R_J
            fixed_contact = fixed_contact[0]
            fixed_contact_regions = True
            # print(f"At iter = {iter}, Fixed contact fixed: {fixed_contact}")
        else:
            fixed_contact_regions = False

        try:
            if fixed_contact_regions == True:
                R, AV, q, u, gNdot, gammaF, J = self.get_R_J(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,fixed_contact)
            else:
                R, AV, q, u, gNdot, gammaF, J, contacts_nu = self.get_R_J(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)
                contacts = np.zeros((MAXITERn+1,3*self.nN+2*self.nN),dtype=int)
                contacts[nu,:] = contacts_nu
            norm_R = np.linalg.norm(R,np.inf)
            print(f"iter = {iter}. nu = {nu}")
            print(f"norm(R) = {norm_R}")

            while np.abs(np.linalg.norm(R,np.inf))>self.tol_n and nu<MAXITERn:
                # Newton Update
                X = X-np.linalg.solve(J,R)
                # Calculate new EOM and residual
                nu = nu+1
                if fixed_contact_regions:
                    R, AV, q, u, gNdot, gammaF, J = self.get_R_J(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,fixed_contact)
                else:
                    R, AV, q, u, gNdot, gammaF, J, contacts_nu = self.get_R_J(X,prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)
                    contacts[nu,:] = contacts_nu
                norm_R = np.linalg.norm(R,np.inf)
                print(f"nu = {nu}")
                print(f"norm(R) = {norm_R}")

                if norm_R>10**9:
                    # the Jacobian is blowing up
                    # (I am assuming this is happening because contact region is fixed, 
                    # update is being called from within solve_bifuration)
                    raise MaxNewtonIterAttainedError
            
            if nu == MAXITERn:
                # print(f"No Convergence at iter = {iter} for nu = {nu} at rho_inf = {rho_inf}")
                raise MaxNewtonIterAttainedError

        except MaxNewtonIterAttainedError as e:
            if fixed_contact_regions is False:

                unique_contacts = np.unique(contacts, axis=0)
                do_not_unpack = True  
                if np.shape(unique_contacts)[0]:
                    return unique_contacts, do_not_unpack
                else:
                    # if unique contact regions were already determined, don't recalculate them
                    unique_A = np.unique(contacts[:,0:self.nN], axis=0)
                

                    # print(f"At iter = {iter}, Max Newton iterations is attained. Unique A contacts are {unique_A}.")

                    unique_contacts = np.empty((0, 10))

                    if np.any(np.all(unique_A == np.array([0,0]), axis=1)):    # check if [0,0] is in 'A'
                        unique_contacts = np.vstack([unique_contacts,self.unique_contacts_a])
                    if np.any(np.all(unique_A == np.array([1,0]), axis=1)):    # check if [1,0] is in 'A'
                        unique_contacts = np.vstack([unique_contacts,self.unique_contacts_b])
                    if np.any(np.all(unique_A == np.array([0,1]), axis=1)):    # check if [0,1] is in 'A'
                        unique_contacts = np.vstack([unique_contacts,self.unique_contacts_c])
                    if np.any(np.all(unique_A == np.array([1,1]), axis=1)):    # check if [1,1] is in 'A'
                        unique_contacts = np.vstack([unique_contacts,self.unique_contacts_d])

                    # because if the number of contact regions is 6 which is the original number
                    # of outputs of update, each row of unique contacts will be assinged as an output variable
                    # if n_tau/int(1/tol_n) <100: # don't keep incrementing infinitely. without the if statement, anytime you don't converge, and you increase rho_inf, you will increase n_tau
                    #     n_tau = n_tau*10
    
                    return unique_contacts, do_not_unpack
            return 
        except np.linalg.LinAlgError as e:
            if norm_R>10**9:
                # the Jacobian is blowing up
                # (I am assuming this is happening because contact region is fixed, 
                # update is being called from within solve_bifuration)
                raise TypeError
            else:
                # the Jacobian matrix is singular, not invertable
                print(e)
                # increment rho_inf        
                self.update_rho_inf()
                # calling function recursively
                self.update(prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF,fixed_contact)
        except Exception as e:
            # any other exception
            raise e
        
        return X,AV,q,u,gNdot,gammaF

    def update_rho_inf(self):
        '''Update the numerical parameter rho_inf.'''
        self.rho_inf = self.rho_inf+0.05  #0.01
        print(self.rho_inf)
        if np.abs(self.rho_inf - self.rho_infinity_initial) < 0.001:
            print("possibility of infinite loop")
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
        lambda_N = X[3*self.ndof+3*ng+2*self.ngamma+2*self.nN:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN]
        Lambda_F = X[3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+self.nF]
        lambda_F = X[3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+self.nF:3*self.ndof+3*self.ng+2*self.ngamma+3*self.nN+2*self.nF]
        return a,U,Q,Kappa_g,Lambda_g,lambda_g,Lambda_gamma,lambda_gamma,\
            Kappa_N,Lambda_N,lambda_N,Lambda_F,lambda_F

    def increment_saved_arrays(self):
        '''Increment saved arrays due to a bifurcation.'''
        
        self.save_arrays()

        # increment saved arrays
        q_save_addition = np.tile(self.q_save[self.leaves_counter,:,:],(1,1,1))
        self.q_save = np.vstack((self.q_save,q_save_addition))
        u_save_addition = np.tile(self.u_save[self.leaves_counter,:,:],(1,1,1))
        self.u_save = np.vstack((self.u_save,u_save_addition))
        X_save_addition = np.tile(self.X_save[self.leaves_counter,:,:],(1,1,1))
        self.X_save = np.vstack((self.X_save,X_save_addition))
        gNdot_save_addition = np.tile(self.gNdot_save[self.leaves_counter,:,:],(1,1,1))
        self.gNdot_save = np.vstack((self.gNdot_save,gNdot_save_addition))
        gammaF_save_addition = np.tile(self.gammaF_save[self.leaves_counter,:,:],(1,1,1))
        self.gammaF_save = np.vstack((self.gammaF_save,gammaF_save_addition))
        AV_save_addition = np.tile(self.AV_save[self.leaves_counter,:,:],(1,1,1))
        self.AV_save = np.vstack((self.AV_save,AV_save_addition))

    def solve_ibi(self,iter_start=1):
        '''Solution iteration by iteration increment.'''

        increment_leaves = True

        # f.write(f'Running solve starting from iteration at leaf {leaves_counter}\n')
        # g.write(f'{iter_start}-')

        prev_X = self.X_save[self.leaves_counter,:,iter_start-1]
        prev_AV = self.AV_save[self.leaves_counter,:,iter_start-1]
        prev_q = self.q_save[self.leaves_counter,:,iter_start-1]
        prev_u = self.u_save[self.leaves_counter,:,iter_start-1]
        prev_gNdot = self.gNdot_save[self.leaves_counter,:,iter_start-1]
        prev_gammaF = self.gammaF_save[self.leaves_counter,:,iter_start-1]
        iter = iter_start
        
        while iter<self.ntime:
            print(f"iteration {iter}")

            try:
                X,AV,q,u,gNdot,gammaF = self.update(prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)
                # this line will return a value error if the MaxNewtonIterAttainedError exception was handeled in update

                prev_X = X
                prev_AV = AV
                prev_q = q
                prev_u = u
                prev_gNdot = gNdot
                prev_gammaF = gammaF

                self.q_save[self.leaves_counter,:,iter] = prev_q
                self.u_save[self.leaves_counter,:,iter] = prev_u
                self.X_save[self.leaves_counter,:,iter] = prev_X
                self.gNdot_save[self.leaves_counter,:,iter] = prev_gNdot
                self.gammaF_save[self.leaves_counter,:,iter] = prev_gammaF
                self.AV_save[self.leaves_counter,:,iter] = prev_AV

                # reset initial value
                self.rho_infinity_initial = self.rho_inf 

            except ValueError as e:
                unique_contacts,_ = self.update(prev_X,prev_AV,prev_q,prev_u,prev_gNdot,prev_gammaF)
                # f.write(f'Detected a bifurcation at leaf {leaves_counter} at iter {iter}\n')
                # g.write(f'{iter}\n')
                self.solve_bifurcation_ibi(iter,unique_contacts)
                increment_leaves = False
                break   # this break is important 
            except Exception as e:
                # f.write(f'Bifurcation branch did not pan out for leaf {leaves_counter} at {iter}\n') 
                raise e

            iter = iter+1

            
            if iter%25 == 0:
                self.save_arrays()

        if increment_leaves == True:

            # g.write(f'end (leaf {leaves_counter})\n')
            
            self.increment_saved_arrays()
            leaves_counter = leaves_counter + 1
            # f.write(f'leaves counter incremented to leaf {leaves_counter}\n')
            # print(f'iter = {iter}. leaves_counter = {leaves_counter}')
            # if leaves_counter>max_leaves:
            #     f.write(f'Program quit because max number of leaves that is {max_leaves} was exceeded.\n')
            #     raise Exception

        return

    def solve_bifurcation_ibi(self,iter_bif,leaves_counter,*fixed_contact_region_params):
        self.bif_counter +=1

        # fixed_contact_regions = True
        unique_contacts = fixed_contact_region_params[0]
        n_unique_contacts = np.shape(unique_contacts)[0]

        nonconvergence_counter = 0

        for k in range(n_unique_contacts):
            iter = iter_bif

            print(f"k = {k}")
            # g.write("     |"*bif_counter)
            # g.write(f'__ {k+1} of {n_unique_contacts}  ')
            
            try:
                fixed_contact  = unique_contacts[k,:]
                X,AV,q,u,gNdot,gammaF = self.update(self.X_save[leaves_counter,:,iter_bif-1],self.AV_save[leaves_counter,:,iter_bif-1],
                                        self.q_save[leaves_counter,:,iter_bif-1],self.u_save[leaves_counter,:,iter_bif-1],
                                        self.gNdot_save[leaves_counter,:,iter_bif-1],self.gammaF_save[leaves_counter,:,iter_bif-1],
                                        fixed_contact)

                self.q_save[leaves_counter,:,iter_bif] = q
                self.u_save[leaves_counter,:,iter_bif] = u
                self.X_save[leaves_counter,:,iter_bif] = X
                self.gNdot_save[leaves_counter,:,iter_bif] = gNdot
                self.gammaF_save[leaves_counter,:,iter_bif] = gammaF
                self.AV_save[leaves_counter,:,iter_bif] = AV
                # f.write(f'{k}-th unique contact convergence successfull\n')            

                self.solve_ibi(iter_bif+1)

                if leaves_counter > self.max_leaves:
                    break

            except TypeError as e:       
                # make a provision for if we always passed and never converged.
                # f.write(f'{k}-th unique contact convergence unsuccessfull\n') 
                # g.write('unsuccessful\n')   
                nonconvergence_counter = nonconvergence_counter+1
                # f.write(f'nonconvergence_counter = {nonconvergence_counter}\n')
                if nonconvergence_counter == n_unique_contacts:
                    # exception raised when None of the fixed contact regions converged
                    # raise Exception
                    nonconvergence_counter = 0
                    if self.MAXITERn < 10:  # INCOMPLETE, TO BE FIXED
                        # try to increase number of iterations
                        self.MAXITERn = 200
                        self.solve_bifurcation(iter_bif,unique_contacts)
                    else:
                        try:
                            self.update_rho_inf()
                            self.solve_bifurcation(iter_bif,unique_contacts)
                        except:
                            # we cannot update rho_inf anymore
                            # we need to abandon this leaf  
                            # g.write(f'bifurcation convergence failed\n')
                            pass
                            raise Exception
                    # solve_bifurcation(iter_bif,unique_contacts) # maybe wrong, remove
                else:
                    pass

        bif_counter = bif_counter-1
        # increment_leaves = False

        return 
    
    def solve_bbb(self):
        return
    
    def solve_bifurcation_bbb(self):
        return


# Test ibi
testibi = Simulation(ntime = 5, mu_s=1, mu_k=0.3, eN=0, eF=0, max_leaves=5)
testibi.solve_ibi()