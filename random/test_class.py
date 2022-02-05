from casadi import *
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


class LCS_learner:
    def __init__(self, n_state, n_lam, A=None, C=None, D=None, G=None, lcp_offset=None, stiffness=0.):
        self.n_lam = n_lam
        self.n_state = n_state

        self.lam = SX.sym('lam', self.n_lam)
        self.x = SX.sym('x', self.n_state)

        self.theta = []

        if G is None:
            self.G = SX.sym('G', self.n_lam, self.n_lam)
            self.theta += [vec(self.G)]
        else:
            self.G = DM(G)

        if D is None:
            self.D = SX.sym('D', self.n_lam, self.n_state)
            self.theta += [vec(self.D)]
        else:
            self.D = DM(D)

        if lcp_offset is None:
            self.lcp_offset = SX.sym('lcp_offset', self.n_lam)
            self.theta += [vec(self.lcp_offset)]
        else:
            self.lcp_offset = DM(lcp_offset)

        if A is None:
            self.A = SX.sym('A', self.n_state, self.n_state)
            self.theta += [vec(self.A)]
        else:
            self.A = DM(A)

        if C is None:
            self.C = SX.sym('C', self.n_state, self.n_lam)
            self.theta += [vec(self.C)]
        else:
            self.C = DM(C)

        self.theta = vcat(self.theta)
        self.n_theta = self.theta.numel()

        self.F = stiffness * np.eye(self.n_lam) + self.G @ self.G.T
        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.G_fn = Function('G_fn', [self.theta], [self.G])
        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.C_fn = Function('C_fn', [self.theta], [self.C])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

    def differetiable(self, gamma=1e-4, epsilon=1e0):

        # define the dynamics loss
        self.x_next = SX.sym('x_next', self.n_state)
        data = vertcat(self.x, self.x_next)
        dyn = self.A @ self.x + self.C @ self.lam
        dyn_loss = dot(dyn - self.x_next, dyn - self.x_next)

        # lcp loss
        dist = self.D @ self.x + self.F @ self.lam + self.lcp_offset
        self.phi = SX.sym('phi', self.n_lam)
        lcp_loss = dot(self.lam, self.phi) + 1 / gamma * dot(self.phi - dist, self.phi - dist)
        self.dist_fn = Function('dist_fn', [self.x, self.lam, self.theta], [dist])

        # total loss
        loss = dyn_loss + lcp_loss / epsilon

        # establish the qp solver
        lam_phi = vertcat(self.lam, self.phi)
        data_theta = vertcat(self.x, self.x_next, self.theta)
        quadprog = {'x': vertcat(self.lam, self.phi), 'f': loss, 'p': data_theta}
        opts = {'printLevel': 'none', }
        self.inner_QPSolver = qpsol('inner_QPSolver', 'qpoases', quadprog, opts)

        # compute the jacobian from lam to theta
        self.loss_fn = Function('loss_fn', [data, self.theta, lam_phi], [loss])
        self.dloss_fn = Function('dloss_fn', [data, self.theta, lam_phi], [jacobian(loss, self.theta).T])
        self.dyn_loss_fn = Function('dyn_loss_fn', [data, self.theta, lam_phi], [dyn_loss])
        self.lcp_loss_fn = Function('lcp_loss_fn', [data, self.theta, lam_phi], [lcp_loss])

        # compute the second order derivative
        grad_loss = jacobian(loss, lam_phi).T
        L = diag(lam_phi) @ grad_loss
        self.L_fn = Function('L_fn', [data, self.theta, lam_phi], [L])  # this is just for testing
        # compute the gradient of lam_phi_opt with respect to theta
        dL_dsol = jacobian(L, lam_phi)
        dL_dtheta = jacobian(L, self.theta)
        dsol_dtheta = -inv(dL_dsol) @ dL_dtheta
        self.dsol_dtheta_fn = Function('dsol_dtheta_fn', [data, self.theta, lam_phi], [dsol_dtheta])
        # this is just for testing
        dloss2 = jacobian(loss, self.theta) + jacobian(loss, lam_phi) @ dsol_dtheta
        self.dloss2_fn = Function('dloss2_fn', [data, self.theta, lam_phi], [dloss2.T])
        # compute the second order derivative
        dloss_dtheta = jacobian(loss, self.theta).T
        ddloss = jacobian(dloss_dtheta, self.theta) + jacobian(dloss_dtheta, lam_phi) @ dsol_dtheta
        self.ddloss_fn = Function('ddloss_fn', [data, self.theta, lam_phi], [ddloss])

    def compute_lambda(self, x_batch, x_next_batch, theta_val):
        self.differetiable()

        # prepare the data
        batch_size = x_batch.shape[0]
        data_batch = np.hstack((x_batch, x_next_batch))
        theta_val_batch = np.tile(theta_val, (batch_size, 1))
        data_theta_batch = np.hstack((data_batch, theta_val_batch))

        # compute the lam_phi solution
        sol_batch = self.inner_QPSolver(lbx=0.0, p=data_theta_batch.T)
        loss_opt_batch = sol_batch['f'].full().flatten()
        lam_phi_opt_batch = sol_batch['x'].full().T

        return lam_phi_opt_batch, loss_opt_batch

    def gradient_step(self, x_batch, x_next_batch, theta_val, lam_phi_opt_batch, second_order=False):
        batch_size = x_batch.shape[0]
        data_batch = np.hstack((x_batch, x_next_batch))
        theta_val_batch = np.tile(theta_val, (batch_size, 1))

        # compute the gradient value
        dtheta_batch = self.dloss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
        dtheta_mean = dtheta_batch.full().mean(axis=1)

        # compute the losses
        loss_batch = self.loss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
        dyn_loss_batch = self.dyn_loss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
        lcp_loss_batch = self.lcp_loss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
        loss_mean = loss_batch.full().mean()
        dyn_loss_mean = dyn_loss_batch.full().mean()
        lcp_loss_mean = lcp_loss_batch.full().mean()

        # this is for testing
        # testing = self.L_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T).full()
        # dtheta_batch2 = self.dloss2_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
        # dtheta_mean2 = dtheta_batch2.full().mean(axis=1)
        # print(norm_2(dtheta_mean-dtheta_mean2))
        dtheta_hessian = dtheta_mean

        if second_order is True:
            hessian_batch = self.ddloss_fn(data_batch.T, theta_val_batch.T, lam_phi_opt_batch.T)
            # compute the mean hessian
            hessian_sum = 0
            for i in range(batch_size):
                hessian_i = hessian_batch[:, i * self.n_theta:(i + 1) * self.n_theta]
                hessian_sum += hessian_i
            hessian_mean = hessian_sum / batch_size
            damping_factor = 1
            u, s, vh = np.linalg.svd(hessian_mean)
            s = s + damping_factor
            damped_hessian = u @ np.diag(s) @ vh
            dtheta_hessian = (inv(damped_hessian) @ DM(dtheta_mean)).full().flatten()

        return dtheta_mean, loss_mean, dyn_loss_mean, lcp_loss_mean, dtheta_hessian

    def dyn_prediction(self, x_batch, theta_val):
        self.differetiable()

        batch_size = x_batch.shape[0]
        theta_val_batch = np.tile(theta_val, (batch_size, 1))
        x_theta_batch = np.hstack((x_batch, theta_val_batch))

        # establish the lcp solver
        dist = self.D @ self.x + self.F @ self.lam
        lcp_loss = dot(dist, self.lam)
        x_theta = vertcat(self.x, self.theta)
        quadprog = {'x': self.lam, 'f': lcp_loss, 'g': dist, 'p': x_theta}
        opts = {'printLevel': 'none'}
        lcp_Solver = qpsol('lcp_solver', 'qpoases', quadprog, opts)

        # establish the dynamics equation
        dyn = self.A @ self.x + self.C @ self.lam
        dyn_fn = Function('dyn_fn', [self.x, self.lam, self.theta], [dyn])

        # compute the lam_batch
        sol_batch = lcp_Solver(lbx=0., lbg=0., p=x_theta_batch.T)
        lam_opt_batch = sol_batch['x'].full().T

        # compute the next state batch
        x_next_batch = dyn_fn(x_batch.T, lam_opt_batch.T, theta_val_batch.T).full().T

        return x_next_batch, lam_opt_batch


# do statistics for the modes
def statiModes(lam_batch, tol=1e-5):
    # dimension of the lambda
    n_lam = lam_batch.shape[1]
    # total number of modes
    total_n_mode = float(2 ** n_lam)

    # do the statistics for the modes
    lam_batch_mode = np.where(lam_batch < tol, 0, 1)
    unique_mode_list, mode_count_list = np.unique(lam_batch_mode, axis=0, return_counts=True)
    mode_frequency_list = mode_count_list / lam_batch.shape[0]

    active_mode_frequence = unique_mode_list.shape[0] / total_n_mode
    # print(active_mode_frequence, total_n_mode)

    return active_mode_frequence, unique_mode_list, mode_frequency_list


# compute the boundaries for each mode
# F \lambda +D x+lcp_offset
# output a polytope for such specified mode: Ax+b>=0
def mode_polytope(F, D, lcp_offset, mode_vector):
    # detect the active mode index
    tol = 1e-6
    active_index = []
    inactive_index = []
    for i in range(mode_vector.size):
        if mode_vector[i] < tol:
            inactive_index += [i]
        else:
            active_index += [i]

    # partition the matrix of F and matrix D into active parts and inactive parts
    F11 = F[active_index][:, active_index]
    F12 = F[active_index][:, inactive_index]
    F21 = F[inactive_index][:, active_index]
    F22 = F[inactive_index][:, inactive_index]
    D1 = D[active_index]
    D2 = D[inactive_index]
    lcp_offset_1 = lcp_offset[active_index]
    lcp_offset_2 = lcp_offset[inactive_index]

    # output
    A = D2 - F21 @ la.inv(F11) @ D1
    b = lcp_offset_2 - F21 @ la.inv(F11) @ lcp_offset_1

    return A, b


# do the plot of differnet mode
def plotModes(lam_batch, tol=1e-5):
    # do the statistics for the modes
    lam_batch_mode = np.where(lam_batch < tol, 0, 1)
    unique_mode_list, mode_indices = np.unique(lam_batch_mode, axis=0, return_inverse=True)

    return unique_mode_list, mode_indices
