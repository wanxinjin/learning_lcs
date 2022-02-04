import time
import casadi
import numpy as np
from casadi import *
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from lcs import SolveLCP
import numpy.linalg as la


class QP_learner:
    def __init__(self, n_state, n_control, n_lam):
        self.n_lam = n_lam
        self.n_state = n_state
        self.n_control = n_control

        self.H = SX.sym('H', self.n_lam, self.n_lam)
        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.F = np.eye(self.n_lam) + self.H @ self.H.T
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.lam = SX.sym('lam', self.n_lam)
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)
        self.theta = vertcat(vec(self.H), vec(self.D), vec(self.E), self.lcp_offset)
        self.n_theta = self.theta.numel()

        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.H_fn = Function('H_fn', [self.theta], [self.H])
        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

    def diff(self):
        # set up the qp solver
        f = dot(self.lam, self.F @ self.lam) + dot(self.lam, self.D @ self.x + self.E @ self.u + self.lcp_offset)
        constraint = self.F @ self.lam + self.D @ self.x + self.E @ self.u + self.lcp_offset
        xu_theta = vertcat(vec(self.x), vec(self.u), self.theta)
        quadprog = {'x': self.lam, 'f': f, 'p': xu_theta, 'g': constraint}
        opts = {'printLevel': 'none', }
        self.qpSolver = qpsol('qpSolver', 'qpoases', quadprog, opts)

        # set the differentiable qp solver
        lam_goal = SX.sym('lam_goal', self.n_lam)
        loss = dot(lam_goal - self.lam, lam_goal - self.lam)
        dloss_dlam = jacobian(loss, self.lam)
        g = diag(self.lam) @ (self.F @ self.lam + self.D @ self.x + self.E @ self.u + self.lcp_offset)
        dg_dtheta = jacobian(g, self.theta)
        dg_dlam = jacobian(g, self.lam)
        dlam_dtehta = -inv(dg_dlam) @ dg_dtheta
        dloss_dtheta = dloss_dlam @ dlam_dtehta
        self.dtheta_fn = Function('dtheta_fn', [self.theta, self.x, self.u, self.lam, lam_goal], [loss, dloss_dtheta.T])

        # test
        self.test_fn = Function('dtheta_fn2', [self.theta, self.x, self.u, self.lam, lam_goal], [norm_2(g)])

    def qpSol(self, theta, x_batch, u_batch):

        self.diff()

        theta = DM(theta)

        if x_batch.ndim == 1:
            x_batch = DM(x_batch)
            u_batch = DM(u_batch)
        else:
            x_batch = DM(x_batch).T
            u_batch = DM(u_batch).T

        batch_size = x_batch.shape[1]

        theta_batch = np.tile(theta, (1, batch_size))
        xu_theta_batch = vertcat(x_batch, u_batch, theta_batch)

        lam_sol = self.qpSolver(p=xu_theta_batch, lbx=0, lbg=0.)

        lam_batch = lam_sol['x'].full().T
        if batch_size == 1:
            lam_batch = lam_batch.flatten()

        return lam_batch

    def gradient(self, theta, x_batch, u_batch, lam_batch, goal_lam_batch):

        theta = DM(theta)

        loss_batch, dtheta_batch = self.dtheta_fn(theta, x_batch.T, u_batch.T, lam_batch.T, goal_lam_batch.T)
        mean_loss = loss_batch.full().mean()
        mean_dtheta = dtheta_batch.full().mean(axis=1)

        res = self.test_fn(theta, x_batch.T, u_batch.T, lam_batch.T, goal_lam_batch.T)
        return mean_loss, mean_dtheta, res.full().mean()


class QPL_learner:
    def __init__(self, n_state, n_control, n_lam):
        self.n_lam = n_lam
        self.n_state = n_state
        self.n_control = n_control

        self.A = SX.sym('A', self.n_state, self.n_state)
        self.B = SX.sym('B', self.n_state, self.n_control)
        self.C = SX.sym('C', self.n_state, self.n_lam)
        self.dyn_offset = SX.sym('dyn_offset', self.n_state)

        self.H = SX.sym('H', self.n_lam, self.n_lam)
        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.F = np.eye(self.n_lam) + self.H @ self.H.T
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.theta = vertcat(vec(self.A), vec(self.B), vec(self.C), self.dyn_offset,
                             vec(self.H), vec(self.D), vec(self.E), self.lcp_offset)
        self.n_theta = self.theta.numel()

        self.lam = SX.sym('lam', self.n_lam)
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)

        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.B_fn = Function('B_fn', [self.theta], [self.B])
        self.C_fn = Function('C_fn', [self.theta], [self.C])
        self.dyn_offset_fn = Function('dyn_offset_fn', [self.theta], [self.dyn_offset])

        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.H_fn = Function('H_fn', [self.theta], [self.H])
        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

    def diff(self):
        # set up the qp solver
        f = dot(self.lam, self.F @ self.lam) + dot(self.lam, self.D @ self.x + self.E @ self.u + self.lcp_offset)
        constraint = self.F @ self.lam + self.D @ self.x + self.E @ self.u + self.lcp_offset
        xu_theta = vertcat(vec(self.x), vec(self.u), self.theta)
        quadprog = {'x': self.lam, 'f': f, 'p': xu_theta, 'g': constraint}
        opts = {'printLevel': 'none', }
        self.qpSolver = qpsol('qpSolver', 'qpoases', quadprog, opts)

        # set the differentiable qp solver
        g = diag(self.lam) @ (self.F @ self.lam + self.D @ self.x + self.E @ self.u + self.lcp_offset)
        dg_dtheta = jacobian(g, self.theta)
        dg_dlam = jacobian(g, self.lam)
        dlam_dtehta = -inv(dg_dlam) @ dg_dtheta

        # define the loss function
        lam_goal = SX.sym('lam_goal', self.n_lam)
        x_goal = SX.sym('x_goal', self.n_state)
        pred_x = self.A @ self.x + self.B @ self.u + self.C @ self.lam + self.dyn_offset
        loss = dot(x_goal - pred_x, x_goal - pred_x)
        dloss_dlam = jacobian(loss, self.lam)
        dloss_dtheta = dloss_dlam @ dlam_dtehta + jacobian(loss, self.theta)
        self.dtheta_fn = Function('dtheta_fn', [self.theta, self.x, self.u, self.lam, x_goal], [loss, dloss_dtheta.T])
        self.dyn_pred_fn = Function('dyn_pred_fn', [self.theta, self.x, self.u, self.lam], [pred_x])

        # test
        self.test_fn = Function('dtheta_fn2', [self.theta, self.x, self.u, self.lam, x_goal], [norm_2(g)])

    def qpSol(self, theta, x_batch, u_batch):

        self.diff()

        theta = DM(theta)

        if x_batch.ndim == 1:
            x_batch = DM(x_batch)
            u_batch = DM(u_batch)
        else:
            x_batch = DM(x_batch).T
            u_batch = DM(u_batch).T

        batch_size = x_batch.shape[1]

        theta_batch = np.tile(theta, (1, batch_size))
        xu_theta_batch = vertcat(x_batch, u_batch, theta_batch)

        lam_sol = self.qpSolver(p=xu_theta_batch, lbx=0, lbg=0.)

        lam_batch = lam_sol['x'].full().T
        if batch_size == 1:
            lam_batch = lam_batch.flatten()

        return lam_batch

    def gradient(self, theta, x_batch, u_batch, lam_batch, goal_x_batch):

        theta = DM(theta)

        loss_batch, dtheta_batch = self.dtheta_fn(theta, x_batch.T, u_batch.T, lam_batch.T, goal_x_batch.T)
        mean_loss = loss_batch.full().mean()
        mean_dtheta = dtheta_batch.full().mean(axis=1)

        res = self.test_fn(theta, x_batch.T, u_batch.T, lam_batch.T, goal_x_batch.T)
        return mean_loss, mean_dtheta, res.full().mean()

    def dyn_pred(self, theta, x_batch, u_batch, lam_batch):
        theta = DM(theta)

        pred_x_batch = self.dyn_pred_fn(theta, x_batch.T, u_batch.T, lam_batch.T)

        pred_x_batch = pred_x_batch.T.full()

        return pred_x_batch


class QPFix_learner:
    def __init__(self, n_state, n_control, n_lam, A, B, C, dyn_offset):
        self.n_lam = n_lam
        self.n_state = n_state
        self.n_control = n_control

        self.A = DM(A)
        self.B = DM(B)
        self.C = DM(C)
        self.dyn_offset = DM(dyn_offset)

        self.H = SX.sym('H', self.n_lam, self.n_lam)
        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.F = np.eye(self.n_lam) + self.H @ self.H.T
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.theta = vertcat(vec(self.H), vec(self.D), vec(self.E), self.lcp_offset)
        self.n_theta = self.theta.numel()

        self.lam = SX.sym('lam', self.n_lam)
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)

        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.B_fn = Function('B_fn', [self.theta], [self.B])
        self.C_fn = Function('C_fn', [self.theta], [self.C])
        self.dyn_offset_fn = Function('dyn_offset_fn', [self.theta], [self.dyn_offset])

        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.H_fn = Function('H_fn', [self.theta], [self.H])
        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

    def diff(self):
        # set up the qp solver
        f = dot(self.lam, self.F @ self.lam) + dot(self.lam, self.D @ self.x + self.E @ self.u + self.lcp_offset)
        constraint = self.F @ self.lam + self.D @ self.x + self.E @ self.u + self.lcp_offset
        xu_theta = vertcat(vec(self.x), vec(self.u), self.theta)
        quadprog = {'x': self.lam, 'f': f, 'p': xu_theta, 'g': constraint}
        opts = {'printLevel': 'none', }
        self.qpSolver = qpsol('qpSolver', 'qpoases', quadprog, opts)

        # set the differentiable qp solver
        g = diag(self.lam) @ (self.F @ self.lam + self.D @ self.x + self.E @ self.u + self.lcp_offset)
        dg_dtheta = jacobian(g, self.theta)
        dg_dlam = jacobian(g, self.lam)
        dlam_dtehta = -inv(dg_dlam) @ dg_dtheta

        # define the loss function
        lam_goal = SX.sym('lam_goal', self.n_lam)
        x_goal = SX.sym('x_goal', self.n_state)
        pred_x = self.A @ self.x + self.B @ self.u + self.C @ self.lam + self.dyn_offset
        loss = dot(x_goal - pred_x, x_goal - pred_x)
        dloss_dlam = jacobian(loss, self.lam)
        dloss_dtheta = dloss_dlam @ dlam_dtehta + jacobian(loss, self.theta)
        self.dtheta_fn = Function('dtheta_fn', [self.theta, self.x, self.u, self.lam, x_goal], [loss, dloss_dtheta.T])
        self.dyn_pred_fn = Function('dyn_pred_fn', [self.theta, self.x, self.u, self.lam], [pred_x])

        # test
        self.test_fn = Function('dtheta_fn2', [self.theta, self.x, self.u, self.lam, x_goal], [norm_2(g)])

    def qpSol(self, theta, x_batch, u_batch):

        self.diff()

        theta = DM(theta)

        if x_batch.ndim == 1:
            x_batch = DM(x_batch)
            u_batch = DM(u_batch)
        else:
            x_batch = DM(x_batch).T
            u_batch = DM(u_batch).T

        batch_size = x_batch.shape[1]

        theta_batch = np.tile(theta, (1, batch_size))
        xu_theta_batch = vertcat(x_batch, u_batch, theta_batch)

        lam_sol = self.qpSolver(p=xu_theta_batch, lbx=0, lbg=0.)

        lam_batch = lam_sol['x'].full().T
        if batch_size == 1:
            lam_batch = lam_batch.flatten()

        return lam_batch

    def gradient(self, theta, x_batch, u_batch, lam_batch, goal_x_batch):

        theta = DM(theta)

        loss_batch, dtheta_batch = self.dtheta_fn(theta, x_batch.T, u_batch.T, lam_batch.T, goal_x_batch.T)
        mean_loss = loss_batch.full().mean()
        mean_dtheta = dtheta_batch.full().mean(axis=1)

        res = self.test_fn(theta, x_batch.T, u_batch.T, lam_batch.T, goal_x_batch.T)
        return mean_loss, mean_dtheta, res.full().mean()

    def dyn_pred(self, theta, x_batch, u_batch, lam_batch):
        theta = DM(theta)

        pred_x_batch = self.dyn_pred_fn(theta, x_batch.T, u_batch.T, lam_batch.T)

        pred_x_batch = pred_x_batch.T.full()

        return pred_x_batch


class QPJoint_learner:
    def __init__(self, n_state, n_control, n_lam):
        self.n_lam = n_lam
        self.n_state = n_state
        self.n_control = n_control

        self.A = SX.sym('A', self.n_state, self.n_state)
        self.B = SX.sym('B', self.n_state, self.n_control)
        self.C = SX.sym('C', self.n_state, self.n_lam)
        self.dyn_offset = SX.sym('dyn_offset', self.n_state)

        self.H = SX.sym('H', self.n_lam, self.n_lam)
        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.F = np.eye(self.n_lam) + self.H @ self.H.T
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.theta = vertcat(vec(self.A), vec(self.B), vec(self.C), self.dyn_offset,
                             vec(self.H), vec(self.D), vec(self.E), self.lcp_offset)
        self.n_theta = self.theta.numel()

        self.lam = SX.sym('lam', self.n_lam)
        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)
        self.next_x = SX.sym('next_x', self.n_state)

        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.B_fn = Function('B_fn', [self.theta], [self.B])
        self.C_fn = Function('C_fn', [self.theta], [self.C])
        self.dyn_offset_fn = Function('dyn_offset_fn', [self.theta], [self.dyn_offset])

        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.H_fn = Function('H_fn', [self.theta], [self.H])
        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

    def diff(self):
        # set up the qp solver
        f = dot(self.lam, self.F @ self.lam) + dot(self.lam, self.D @ self.x + self.E @ self.u + self.lcp_offset) + \
            dot(self.A @ self.x + self.B @ self.u + self.C @ self.lam + self.dyn_offset - self.next_x,
                self.A @ self.x + self.B @ self.u + self.C @ self.lam + self.dyn_offset - self.next_x)
        constraint = vertcat(self.F @ self.lam + self.D @ self.x + self.E @ self.u + self.lcp_offset, self.lam)
        xu_theta = vertcat(vec(self.x), vec(self.u), self.theta)
        lam_nextx = vertcat(self.lam, self.next_x)
        quadprog = {'x': lam_nextx, 'f': f, 'p': xu_theta, 'g': constraint}
        opts = {'printLevel': 'none', }
        self.qpSolver = qpsol('qpSolver', 'qpoases', quadprog, opts)

        # set the differentiable qp solver
        lam_constraint = SX.sym('lam_constraint', self.n_lam + self.n_lam)
        L = f + dot(lam_constraint, constraint)
        g = jacobian(L, lam_nextx).T
        dg_dtheta = jacobian(g, self.theta)
        dg_dlam_nextx = jacobian(g, lam_nextx)
        dlam_nextx_dtehta = -inv(dg_dlam_nextx) @ dg_dtheta

        # define the loss function
        lam_goal = SX.sym('lam_goal', self.n_lam)
        x_goal = SX.sym('x_goal', self.n_state)
        loss = dot(x_goal - self.next_x, x_goal - self.next_x)
        dloss_dlam_nextx = jacobian(loss, lam_nextx)
        dloss_dtheta = dloss_dlam_nextx @ dlam_nextx_dtehta + jacobian(loss, self.theta)
        self.dtheta_fn = Function('dtheta_fn', [self.theta, self.x, self.u, lam_nextx, lam_constraint, x_goal],
                                  [loss, dloss_dtheta.T])

        # test
        self.test_fn = Function('dtheta_fn2', [self.theta, self.x, self.u, lam_nextx, lam_constraint, x_goal],
                                [norm_2(g)])

    def qpSol(self, theta, x_batch, u_batch):

        self.diff()

        theta = DM(theta)

        if x_batch.ndim == 1:
            x_batch = DM(x_batch)
            u_batch = DM(u_batch)
        else:
            x_batch = DM(x_batch).T
            u_batch = DM(u_batch).T

        batch_size = x_batch.shape[1]

        theta_batch = np.tile(theta, (1, batch_size))
        xu_theta_batch = vertcat(x_batch, u_batch, theta_batch)

        lam_sol = self.qpSolver(p=xu_theta_batch, lbg=0.)

        lam_nextx_batch = lam_sol['x'].full().T
        lam_constraint_batch = lam_sol['lam_g'].full().T
        if batch_size == 1:
            lam_nextx_batch = lam_nextx_batch.flatten()
            lam_constraint_batch = lam_constraint_batch.flatten()

        return lam_nextx_batch, lam_constraint_batch

    def gradient(self, theta, x_batch, u_batch, lam_nextx_batch, lam_constraint_batch, goal_x_batch):

        theta = DM(theta)

        loss_batch, dtheta_batch = self.dtheta_fn(theta, x_batch.T, u_batch.T, lam_nextx_batch.T,
                                                  lam_constraint_batch.T,
                                                  goal_x_batch.T)
        mean_loss = loss_batch.full().mean()
        mean_dtheta = dtheta_batch.full().mean(axis=1)

        res = self.test_fn(theta, x_batch.T, u_batch.T, lam_nextx_batch.T, lam_constraint_batch.T, goal_x_batch.T)
        return mean_loss, mean_dtheta, res.full().mean()


class QPState_learner:
    def __init__(self, n_state, n_control, n_lam):
        self.n_lam = n_state
        self.n_state = n_state
        self.n_control = n_control

        self.A = SX.sym('A', self.n_state, self.n_state)
        self.B = SX.sym('B', self.n_state, self.n_control)
        self.dyn_offset = SX.sym('dyn_offset', self.n_state)
        self.H = SX.sym('H', self.n_control, self.n_control)
        self.F = np.eye(self.n_control) + self.H @ self.H.T

        self.C = SX.sym('C', self.n_lam, self.n_state)
        self.D = SX.sym('D', self.n_lam, self.n_state)
        self.E = SX.sym('E', self.n_lam, self.n_control)
        self.lcp_offset = SX.sym('lcp_offset', self.n_lam)

        self.theta = vertcat(vec(self.A), vec(self.B), vec(self.H), self.dyn_offset,
                             vec(self.C), vec(self.D), vec(self.E), self.lcp_offset)
        self.n_theta = self.theta.numel()

        self.x = SX.sym('x', self.n_state)
        self.u = SX.sym('u', self.n_control)
        self.nextx = SX.sym('next_x', self.n_state)

        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.B_fn = Function('B_fn', [self.theta], [self.B])
        self.C_fn = Function('C_fn', [self.theta], [self.C])
        self.dyn_offset_fn = Function('dyn_offset_fn', [self.theta], [self.dyn_offset])

        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.H_fn = Function('H_fn', [self.theta], [self.H])
        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

    def diff(self):
        # set up the qp solver
        f = dot(self.nextx, self.F @ self.nextx) + dot(self.nextx, self.A @ self.x + self.B @ self.u + self.dyn_offset)
        constraint = self.F @ self.nextx + self.D @ self.x + self.E @ self.u + self.lcp_offset
        xu_theta = vertcat(vec(self.x), vec(self.u), self.theta)
        quadprog = {'x': self.nextx, 'f': f, 'p': xu_theta, 'g': constraint}
        opts = {'printLevel': 'none', }
        self.qpSolver = qpsol('qpSolver', 'qpoases', quadprog, opts)

        # set the differentiable qp solver
        lam = SX.sym('lam', self.n_lam)
        L = f + dot(lam, constraint)
        g = jacobian(L, self.nextx).T
        dg_dtheta = jacobian(g, self.theta)
        dg_dlam_nextx = jacobian(g, self.nextx)
        dnextx_dtehta = -inv(dg_dlam_nextx) @ dg_dtheta

        # define the loss function
        x_goal = SX.sym('x_goal', self.n_state)
        loss = dot(x_goal - self.nextx, x_goal - self.nextx)
        dloss_dnextx = jacobian(loss, self.nextx)
        dloss_dtheta = dloss_dnextx @ dnextx_dtehta + jacobian(loss, self.theta)
        self.dtheta_fn = Function('dtheta_fn', [self.theta, self.x, self.u, self.nextx, lam, x_goal],
                                  [loss, dloss_dtheta.T])

        # test
        self.test_fn = Function('dtheta_fn2', [self.theta, self.x, self.u, self.nextx, lam, x_goal],
                                [norm_2(g)])

    def qpSol(self, theta, x_batch, u_batch):

        self.diff()

        theta = DM(theta)

        if x_batch.ndim == 1:
            x_batch = DM(x_batch)
            u_batch = DM(u_batch)
        else:
            x_batch = DM(x_batch).T
            u_batch = DM(u_batch).T

        batch_size = x_batch.shape[1]

        theta_batch = np.tile(theta, (1, batch_size))
        xu_theta_batch = vertcat(x_batch, u_batch, theta_batch)

        sol = self.qpSolver(p=xu_theta_batch, lbg=0.)

        nextx_batch = sol['x'].full().T
        lam_batch = sol['lam_g'].full().T
        if batch_size == 1:
            nextx_batch = nextx_batch.flatten()
            lam_batch = lam_batch.flatten()

        return nextx_batch, lam_batch

    def gradient(self, theta, x_batch, u_batch, nextx_batch, lam_batch, goal_x_batch):

        theta = DM(theta)

        loss_batch, dtheta_batch = self.dtheta_fn(theta, x_batch.T, u_batch.T, nextx_batch.T,
                                                  lam_batch.T,
                                                  goal_x_batch.T)
        mean_loss = loss_batch.full().mean()
        mean_dtheta = dtheta_batch.full().mean(axis=1)

        res = self.test_fn(theta, x_batch.T, u_batch.T, nextx_batch.T, lam_batch.T, goal_x_batch.T)
        return mean_loss, mean_dtheta, res.full().mean()


class LCS_PN:

    def __init__(self, n_state, n_control, n_lam, A=None, B=None, C=None, dyn_offset=None,
                 D=None, E=None, G=None, H=None, lcp_offset=None, F_stiffness=1.0):
        self.n_state = n_state
        self.n_control = n_control
        self.n_lam = n_lam

        # define system matrices
        self.tunable_para = []
        if A is None:
            self.A = SX.sym('A', n_state, n_state)
            self.tunable_para += [vec(self.A)]
        else:
            self.A = DM(A)

        if B is None:
            self.B = SX.sym('B', n_state, n_control)
            self.tunable_para += [vec(self.B)]
        else:
            self.B = DM(B)

        if C is None:
            self.C = SX.sym('C', n_state, n_lam)
            self.tunable_para += [vec(self.C)]
        else:
            self.C = DM(C)

        if dyn_offset is None:
            self.dyn_offset = SX.sym('dyn_offset', n_state)
            self.tunable_para += [vec(self.dyn_offset)]
        else:
            self.dyn_offset = DM(dyn_offset)

        if D is not None:
            self.D = DM(D)
        else:
            self.D = SX.sym('D', n_lam, n_state)
            self.tunable_para += [vec(self.D)]

        if E is None:
            self.E = SX.sym('E', n_lam, n_control)
            self.tunable_para += [vec(self.E)]
        else:
            self.E = DM(E)

        if G is None:
            self.G = SX.sym('G', n_lam, n_lam)
            self.tunable_para += [vec(self.G)]
        else:
            self.G = DM(G)

        if H is None:
            self.H = SX.sym('H', n_lam, n_lam)
            self.tunable_para += [vec(self.H)]
        else:
            self.H = DM(H)

        if lcp_offset is None:
            self.lcp_offset = SX.sym('lcp_offset', n_lam)
            self.tunable_para += [vec(self.lcp_offset)]
        else:
            self.lcp_offset = DM(lcp_offset)

        self.theta = vcat(self.tunable_para)
        self.n_theta = self.theta.numel()

        self.F = self.G @ self.G.T + F_stiffness * np.eye(self.n_lam) + self.H - self.H.T
        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.G_fn = Function('F_fn', [self.theta], [self.G])
        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.B_fn = Function('B_fn', [self.theta], [self.B])
        self.C_fn = Function('C_fn', [self.theta], [self.C])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.dyn_offset_fn = Function('dyn_offset_fn', [self.theta], [self.dyn_offset])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

    def toMatG(self, G_para):

        if type(G_para) is casadi.SX:
            G = SX(self.n_lam, self.n_lam)
        else:
            G = DM(self.n_lam, self.n_lam)

        for j in range(self.n_lam):
            for i in range(j + 1):
                if i == j:
                    G[i, j] = (G_para[int((j + 1) * j / 2) + i])
                else:
                    G[i, j] = G_para[int((j + 1) * j / 2) + i]
        return G

    def diff(self, F_ref=0., w_F=0., C_ref=0., w_C=0.):
        # define the system variable
        x = casadi.SX.sym('x', self.n_state)
        u = casadi.SX.sym('u', self.n_control)
        xu_pair = vertcat(x, u)
        lam = casadi.SX.sym('lam', self.n_lam)
        x_next = casadi.SX.sym('x_next', self.n_state)

        # dynamics
        dyn = self.A @ x + self.B @ u + self.C @ lam + self.dyn_offset
        self.dyn_fn = Function('dyn_fn', [xu_pair, lam, self.theta], [dyn])

        # lcp function
        lcp_cstr = self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset
        lcp_loss = dot(lam, lcp_cstr)
        self.lcp_loss_fn = Function('dyn_loss_fn', [xu_pair, lam, self.theta], [lcp_loss])
        self.lcp_cstr_fn = Function('dis_cstr_fn', [xu_pair, lam, self.theta], [lcp_cstr])

        # establish the qp solver to solve for lcp
        xu_theta = vertcat(xu_pair, self.theta)
        quadprog = {'x': lam, 'f': lcp_loss, 'g': lcp_cstr, 'p': xu_theta}
        opts = {'printLevel': 'none', }
        self.lcpSolver = qpsol('lcpSolver', 'qpoases', quadprog, opts)

        # differentiate through lcp
        g = diag(lam) @ lcp_cstr
        dg_dlam = jacobian(g, lam)
        dg_dtheta = jacobian(g, self.theta)
        dlam_dtheta = -inv(dg_dlam) @ dg_dtheta

        # compute the gradient of f w.r.t. theta
        pred_loss = (dot(dyn - x_next, dyn - x_next)) \
                    + w_C * dot(vec(DM(C_ref)) - vec(self.C), vec(DM(C_ref)) - vec(self.C)) \
                    + w_F * dot(vec(DM(F_ref)) - vec(self.F), vec(DM(F_ref)) - vec(self.F))

        dtheta = jacobian(pred_loss, self.theta) + jacobian(pred_loss, lam) @ dlam_dtheta
        self.pred_loss_fn = Function('dtheta_fn', [xu_pair, lam, self.theta, x_next], [pred_loss, dtheta.T])
        self.pred_loss_fn2 = Function('dtheta_fn', [xu_pair, lam, self.theta, x_next],
                                      [pred_loss / dot(x_next, x_next), dtheta.T])
        self.dyn_pred = Function('dyn_pred', [xu_pair, lam, self.theta], [dyn])

    def step(self, batch_x, batch_u, batch_x_next, current_theta):

        # solve the LCP equation
        batch_size = batch_x.shape[0]
        xu_batch = np.hstack((batch_x, batch_u)).T
        theta_batch = np.tile(current_theta, (batch_size, 1))
        solver_xutheta_batch = np.hstack((batch_x, batch_u, theta_batch)).T
        sol = self.lcpSolver(p=solver_xutheta_batch, lbx=0., lbg=0.)
        lam_batch = sol['x'].full()

        # solve the prediction loss and the gradient of loss with respect to theta
        pred_loss_batch, dtheta_batch = self.pred_loss_fn(xu_batch, lam_batch,
                                                          current_theta, batch_x_next.T)

        mean_pred_loss = pred_loss_batch.full().mean()
        mean_dtheta = dtheta_batch.full().mean(axis=1)

        return mean_pred_loss, mean_dtheta, lam_batch.T

    def pred_error(self, batch_x, batch_u, batch_x_next, current_theta):

        # solve the LCP equation
        batch_size = batch_x.shape[0]
        xu_batch = np.hstack((batch_x, batch_u)).T
        theta_batch = np.tile(current_theta, (batch_size, 1))
        solver_xutheta_batch = np.hstack((batch_x, batch_u, theta_batch)).T
        sol = self.lcpSolver(p=solver_xutheta_batch, lbx=0., lbg=0.)
        lam_batch = sol['x'].full()

        # solve the prediction loss and the gradient of loss with respect to theta
        pred_loss_batch, dtheta_batch = self.pred_loss_fn(xu_batch, lam_batch,
                                                          current_theta, batch_x_next.T)
        mean_prediction_loss = pred_loss_batch.full().mean()
        mean_x_next_norm = (batch_x_next * batch_x_next).sum(axis=1).mean()
        # print(abs(self.dyn_fn(xu_batch,lam_batch,current_theta).full().T-batch_x_next)/batch_x_next)

        mean_pred_loss_relative = mean_prediction_loss / mean_x_next_norm

        return mean_pred_loss_relative

    def pred(self, batch_x, batch_u, current_theta):

        # do the prediction
        batch_size = batch_x.shape[0]
        xu_batch = np.hstack((batch_x, batch_u)).T
        theta_batch = np.tile(current_theta, (batch_size, 1))
        solver_xutheta_batch = np.hstack((batch_x, batch_u, theta_batch)).T
        sol = self.lcpSolver(p=solver_xutheta_batch, lbx=0., lbg=0.)
        lam_batch = sol['x'].full()

        pred_next_x = self.dyn_pred(xu_batch, lam_batch, current_theta)

        return pred_next_x.T.full(), lam_batch.T


class LCS_VN:

    def __init__(self, n_state, n_control, n_lam, A=None, B=None, C=None, dyn_offset=None,
                 D=None, E=None, G=None, H=None, lcp_offset=None, F_stiffness=1.0):
        self.n_state = n_state
        self.n_control = n_control
        self.n_lam = n_lam

        # define system matrices
        self.tunable_para = []
        if A is None:
            self.A = SX.sym('A', n_state, n_state)
            self.tunable_para += [vec(self.A)]
        else:
            self.A = DM(A)

        if B is None:
            self.B = SX.sym('B', n_state, n_control)
            self.tunable_para += [vec(self.B)]
        else:
            self.B = DM(B)

        if C is None:
            self.C = SX.sym('C', n_state, n_lam)
            self.tunable_para += [vec(self.C)]
        else:
            self.C = DM(C)

        if dyn_offset is None:
            self.dyn_offset = SX.sym('dyn_offset', n_state)
            self.tunable_para += [vec(self.dyn_offset)]
        else:
            self.dyn_offset = DM(dyn_offset)

        if D is not None:
            self.D = DM(D)
        else:
            self.D = SX.sym('D', n_lam, n_state)
            self.tunable_para += [vec(self.D)]

        if E is None:
            self.E = SX.sym('E', n_lam, n_control)
            self.tunable_para += [vec(self.E)]
        else:
            self.E = DM(E)

        if G is None:
            self.G = SX.sym('G', n_lam, n_lam)
            self.tunable_para += [vec(self.G)]
        else:
            self.G = DM(G)

        if H is None:
            self.H = SX.sym('H', n_lam, n_lam)
            self.tunable_para += [vec(self.H)]
        else:
            self.H = DM(H)

        if lcp_offset is None:
            self.lcp_offset = SX.sym('lcp_offset', n_lam)
            self.tunable_para += [vec(self.lcp_offset)]
        else:
            self.lcp_offset = DM(lcp_offset)

        self.theta = vcat(self.tunable_para)
        self.n_theta = self.theta.numel()

        self.F = self.G @ self.G.T + F_stiffness * np.eye(self.n_lam) + self.H - self.H.T
        self.F_fn = Function('F_fn', [self.theta], [self.F])
        self.E_fn = Function('E_fn', [self.theta], [self.E])
        self.G_fn = Function('F_fn', [self.theta], [self.G])
        self.A_fn = Function('A_fn', [self.theta], [self.A])
        self.B_fn = Function('B_fn', [self.theta], [self.B])
        self.C_fn = Function('C_fn', [self.theta], [self.C])
        self.D_fn = Function('D_fn', [self.theta], [self.D])
        self.dyn_offset_fn = Function('dyn_offset_fn', [self.theta], [self.dyn_offset])
        self.lcp_offset_fn = Function('lcp_offset_fn', [self.theta], [self.lcp_offset])

    def diff(self, gamma=1e-2, epsilon=1., F_ref=0., w_F=0., C_ref=0., w_C=0.):

        # define the system variable
        x = SX.sym('x', self.n_state)
        u = SX.sym('u', self.n_control)
        x_next = SX.sym('x_next', self.n_state)
        data_pair = vertcat(x, u, x_next)

        # define the variables for LCP
        lam = SX.sym('lam', self.n_lam)
        phi = SX.sym('phi', self.n_lam)
        lam_phi = vertcat(lam, phi)

        # dynamics
        dyn = self.A @ x + self.B @ u + self.C @ lam + self.dyn_offset
        dyn_loss = dot(dyn - x_next, dyn - x_next)

        # lcp loss
        dist = self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset
        lcp_aug_loss = dot(lam, phi) + 1 / gamma * dot(phi - dist, phi - dist)

        # define loss function
        total_loss = 1. * dyn_loss \
                     + 1. / epsilon * lcp_aug_loss \
                     + w_C * dot(vec(DM(C_ref)) - vec(self.C), vec(DM(C_ref)) - vec(self.C)) \
                     + w_F * dot(vec(DM(F_ref)) - vec(self.F), vec(DM(F_ref)) - vec(self.F))

        # establish the qp solver
        data_theta = vertcat(data_pair, self.theta)
        quadprog = {'x': lam_phi, 'f': total_loss, 'p': data_theta}
        opts = {'printLevel': 'none', }
        self.qpSolver = qpsol('Solver_QP', 'qpoases', quadprog, opts)

        # compute the jacobian from lam to theta
        mu = SX.sym('mu', self.n_lam + self.n_lam)
        L = total_loss + dot(mu, lam_phi)

        # compute the derivative of the value function with respect to theta using method 2
        self.loss_fn = Function('loss_fn', [data_pair, lam_phi, mu, self.theta],
                                [jacobian(L, self.theta).T, dyn_loss, lcp_aug_loss, total_loss])

        self.test_fn = Function('test_fn', [data_pair, lam, self.theta], [vertcat(lam, dist)])
        self.test_fn2 = Function('loss_fn', [data_pair, lam_phi, self.theta],
                                 [total_loss])

    def compute_lam(self, batch_x, batch_u, batch_x_next, curr_theta):
        # solve for the
        batch_size = batch_x.shape[0]
        theta_batch = np.tile(curr_theta, (batch_size, 1))
        data_theta_batch = np.hstack((batch_x, batch_u, batch_x_next, theta_batch))

        sol = self.qpSolver(lbx=0., p=data_theta_batch.T)
        loss_batch = sol['f'].full()
        lam_phi_batch = sol['x'].full()
        lam_batch = lam_phi_batch[0:self.n_lam, :]
        phi_batch = lam_phi_batch[self.n_lam:, :]
        mu_batch = sol['lam_x'].full()

        return lam_phi_batch.T, mu_batch.T, loss_batch.flatten(), lam_batch.T, phi_batch.T

    def gradient(self, batch_x, batch_u, batch_x_next, batch_lam_phi, batch_mu, curr_theta):
        data_batch = np.hstack((batch_x, batch_u, batch_x_next))

        # solve the gradient
        dtheta_batch, dyn_loss_batch, lcp_loss_batch, total_loss_batch = self.loss_fn(data_batch.T, batch_lam_phi.T,
                                                                                      batch_mu.T,
                                                                                      curr_theta)
        # do the update of gradient descent
        mean_total_loss = total_loss_batch.full().mean()
        mean_dtheta = dtheta_batch.full().mean(axis=1)
        mean_dyn_loss = dyn_loss_batch.full().mean()
        mean_lcp_loss = lcp_loss_batch.full().mean()

        return mean_total_loss, mean_dtheta, mean_dyn_loss, mean_lcp_loss

    def diff2(self, gamma=1e-2, epsilon=1.):

        x = SX.sym('x', self.n_state)
        u = SX.sym('u', self.n_control)
        true_lam = SX.sym('tru_lam', self.n_lam)
        data_pair = vertcat(x, u, true_lam)

        # define the variables for LCP
        lam = SX.sym('lam', self.n_lam)
        phi = SX.sym('phi', self.n_lam)
        lam_phi = vertcat(lam, phi)

        # dynamics
        # C = np.random.randn(4, self.n_lam)
        # C = np.array([[0, 0],
        #               [1, 1]])
        # C=np.eye(self.n_lam)
        # C = np.ones((4, 2))
        C = 10*np.random.rand(4, self.n_lam)
        error_lam = C @ (lam - true_lam)
        dyn_loss = dot(error_lam, error_lam)

        # lcp loss
        dist = self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset
        lcp_aug_loss = dot(lam, phi) + 1 / gamma * dot(phi - dist, phi - dist)

        # define loss function
        total_loss = 1. * dyn_loss \
                     + 1. / epsilon * lcp_aug_loss

        # establish the qp solver
        data_theta = vertcat(data_pair, self.theta)
        quadprog = {'x': lam_phi, 'f': total_loss, 'p': data_theta}
        opts = {'printLevel': 'none', }
        self.qpSolver2 = qpsol('Solver_QP', 'qpoases', quadprog, opts)

        # compute the jacobian from lam to theta
        mu = SX.sym('mu', self.n_lam + self.n_lam)
        L = total_loss + dot(mu, lam_phi)

        # compute the derivative of the value function with respect to theta using method 2
        self.loss_fn2 = Function('loss_fn', [data_pair, lam_phi, mu, self.theta],
                                 [jacobian(L, self.theta).T, dyn_loss, lcp_aug_loss, total_loss])

    def compute_lam2(self, batch_x, batch_u, batch_lam, curr_theta):
        # solve for the
        batch_size = batch_x.shape[0]
        theta_batch = np.tile(curr_theta, (batch_size, 1))
        data_theta_batch = np.hstack((batch_x, batch_u, batch_lam, theta_batch))

        sol = self.qpSolver2(lbx=0., p=data_theta_batch.T)
        opt_loss_batch = sol['f'].full()
        opt_lam_phi_batch = sol['x'].full()
        opt_lam_batch = opt_lam_phi_batch[0:self.n_lam, :]
        opt_phi_batch = opt_lam_phi_batch[self.n_lam:, :]
        opt_mu_batch = sol['lam_x'].full()

        return opt_lam_phi_batch.T, opt_mu_batch.T, opt_loss_batch.flatten(), opt_lam_batch.T, opt_phi_batch.T

    def gradient2(self, batch_x, batch_u, batch_lam, opt_batch_lam_phi, opt_batch_mu, curr_theta):
        data_batch = np.hstack((batch_x, batch_u, batch_lam))

        # solve the gradient
        dtheta_batch, dyn_loss_batch, lcp_loss_batch, total_loss_batch = self.loss_fn2(data_batch.T,
                                                                                       opt_batch_lam_phi.T,
                                                                                       opt_batch_mu.T,
                                                                                       curr_theta)
        # do the update of gradient descent
        mean_total_loss = total_loss_batch.full().mean()
        mean_dtheta = dtheta_batch.full().mean(axis=1)
        mean_dyn_loss = dyn_loss_batch.full().mean()
        mean_lcp_loss = lcp_loss_batch.full().mean()

        return mean_total_loss, mean_dtheta, mean_dyn_loss, mean_lcp_loss


class LCS_Gen:

    def __init__(self, n_state, n_control, n_lam, A=None, B=None, C=None, dyn_offset=None,
                 D=None, E=None, F=None, lcp_offset=None):
        self.n_state = n_state
        self.n_control = n_control
        self.n_lam = n_lam

        self.A = DM(A)
        self.B = DM(B)
        self.C = DM(C)
        self.dyn_offset = DM(dyn_offset)
        self.D = DM(D)
        self.E = DM(E)
        self.F = DM(F)
        self.lcp_offset = DM(lcp_offset)

        # define the system variable
        x = casadi.SX.sym('x', self.n_state)
        u = casadi.SX.sym('u', self.n_control)
        xu_pair = vertcat(x, u)
        lam = casadi.SX.sym('lam', self.n_lam)
        x_next = casadi.SX.sym('x_next', self.n_state)

        # dynamics
        dyn = self.A @ x + self.B @ u + self.C @ lam + self.dyn_offset
        self.dyn_fn = Function('dyn_fn', [xu_pair, lam], [dyn])

        # lcp function
        lcp_cstr = self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset
        lcp_loss = dot(lam, lcp_cstr)
        self.lcp_loss_fn = Function('dyn_loss_fn', [xu_pair, lam], [lcp_loss])
        self.lcp_cstr_fn = Function('dis_cstr_fn', [xu_pair, lam], [lcp_cstr])

        # establish the qp solver to solve for lcp
        xu_theta = vertcat(xu_pair)
        quadprog = {'x': lam, 'f': lcp_loss, 'g': lcp_cstr, 'p': xu_theta}
        opts = {'printLevel': 'none', }
        self.lcpSolver = qpsol('lcpSolver', 'qpoases', quadprog, opts)

    def nextState(self, x_batch, u_batch):
        xu_batch = np.hstack((x_batch, u_batch))

        sol = self.lcpSolver(p=xu_batch.T, lbx=0., lbg=0.)
        lam_batch = sol['x'].full()

        # compute the next state batch
        x_next_batch = self.dyn_fn(xu_batch.T, lam_batch).full()

        return x_next_batch.T, lam_batch.T

    def nextState_IPOPT(self, x_batch, u_batch):
        xu_batch = np.hstack((x_batch, u_batch))

        q_batch = self.D @ x_batch.T + self.E @ u_batch.T + self.lcp_offset
        lam_batch = SolveLCP.SolveLCP_IPOPT(self.F, q_batch.full())
        # compute the next state batch
        x_next_batch = self.dyn_fn(xu_batch.T, lam_batch).full()

        return x_next_batch.T, lam_batch.T

    def nextState_Lemke(self, x_batch, u_batch):
        xu_batch = np.hstack((x_batch, u_batch))

        q_batch = self.D @ x_batch.T + self.E @ u_batch.T + self.lcp_offset
        lam_batch = SolveLCP.SolveLCP_Lemke(self.F, q_batch.full())
        # compute the next state batch
        x_next_batch = self.dyn_fn(xu_batch.T, lam_batch).full()

        return x_next_batch.T, lam_batch.T


# some facility functions
def vec2MatG(G_para, n_lam):
    if type(G_para) is casadi.SX:
        G = SX(n_lam, n_lam)
    else:
        G = DM(n_lam, n_lam)

    for j in range(n_lam):
        for i in range(j + 1):
            if i == j:
                G[i, j] = (G_para[int((j + 1) * j / 2) + i])
            else:
                G[i, j] = G_para[int((j + 1) * j / 2) + i]
    return G


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


# generate a random lcs
def gen_lcs(n_state, n_control, n_lam, stiffness, gb=1.0):
    A = np.random.uniform(-1, 1, size=(n_state, n_state))
    B = np.random.uniform(-1, 1, size=(n_state, n_control))
    C = 1 * np.random.uniform(-1, 1, size=(n_state, n_lam))
    dyn_offset = 1 * np.random.uniform(-1, 1, size=n_state)
    D = np.random.uniform(-1, 1, size=(n_lam, n_state))
    E = np.random.uniform(-1, 1, size=(n_lam, n_control))
    G_para = gb * np.random.uniform(-1, 1, size=int((n_lam + 1) * n_lam / 2))
    G = vec2MatG(G_para, n_lam)
    H = np.random.uniform(-1, 1, size=(n_lam, n_lam))
    F = G @ G.T + H - H.T + stiffness * np.eye(n_lam)
    lcp_offset = 1 * np.random.uniform(-1, 1, size=n_lam)

    min_sig = min(np.linalg.eigvals(F + F.T))

    lsc_theta = veccat(vec(A), vec(B),
                       vec(C),
                       dyn_offset,
                       vec(D), vec(E),
                       vec(G_para),
                       vec(H),
                       lcp_offset,
                       ).full().flatten()

    lcs_mats = {
        'n_state': n_state,
        'n_control': n_control,
        'n_lam': n_lam,
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'dyn_offset': dyn_offset,
        'E': E,
        'G_para': G_para,
        'H': H,
        'F': F,
        'lcp_offset': lcp_offset,
        'theta': lsc_theta,
        'min_sig': min_sig}

    return lcs_mats


# generate the training data
def gen_data(lcs_mats, data_size, noise_level=0.0, algorithm='qp', magnitude_x=0.1, magnitude_u=1):
    n_state = lcs_mats['n_state']
    n_control = lcs_mats['n_control']
    n_lam = lcs_mats['n_lam']
    A = lcs_mats['A']
    B = lcs_mats['B']
    C = lcs_mats['C']
    dyn_offset = lcs_mats['dyn_offset']
    D = lcs_mats['D']
    E = lcs_mats['E']
    F = lcs_mats['F']
    lcp_offset = lcs_mats['lcp_offset']

    # create the data generator
    lcs_expert = LCS_Gen(n_state=n_state, n_control=n_control, n_lam=n_lam,
                         A=A, B=B, C=C, dyn_offset=dyn_offset,
                         D=D, E=E, F=F, lcp_offset=lcp_offset)

    x_batch = magnitude_x * np.random.uniform(-1, 1, size=(data_size, n_state))
    x_batch = x_batch + noise_level * np.random.randn(*x_batch.shape)
    u_batch = magnitude_u * np.random.uniform(-1, 1, size=(data_size, n_control))
    u_batch = u_batch + noise_level * np.random.randn(*u_batch.shape)

    if algorithm == 'qp':
        x_next_batch, lam_batch = lcs_expert.nextState(x_batch=x_batch, u_batch=u_batch)
    elif algorithm == 'ipopt':
        x_next_batch, lam_batch = lcs_expert.nextState_IPOPT(x_batch=x_batch, u_batch=u_batch)
    else:
        x_next_batch, lam_batch = lcs_expert.nextState_Lemke(x_batch=x_batch, u_batch=u_batch)

    x_next_batch = x_next_batch + noise_level * np.random.randn(*x_next_batch.shape)

    # compute the statistics of the modes
    mode_percentage, unique_mode_list, mode_frequency_list = statiModes(lam_batch)

    data = {'x_batch': x_batch,
            'u_batch': u_batch,
            'lam_batch': lam_batch,
            'x_next_batch': x_next_batch,
            'mode_percentage': mode_percentage,
            'unique_mode_list': unique_mode_list,
            'mode_frequence_list': mode_frequency_list}

    return data
