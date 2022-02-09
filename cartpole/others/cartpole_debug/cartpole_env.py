import time

from lcs import lcs_learning
import numpy as np
from casadi import *
import lcs.optim as opt
from lcs import lcs_control
import matplotlib.pyplot as plt


class cartpole_nonlinear:
    def __init__(self):
        self.name = 'nonlinear cartpole system with wall impacts'

        self.g = 9.81
        self.mp = 0.411
        self.mc = 0.978
        self.len_p = 0.6
        self.len_com = 0.4267
        self.d1 = 0.35
        self.d2 = -0.35
        self.ks = 50
        self.Ts = 0.01

        self.n_state = 4
        self.n_control = 1
        self.n_lam = 2

        # the following is a lcp solver
        F = SX.sym('F', self.n_lam, self.n_lam)
        q = SX.sym('q', self.n_lam)
        para_vec = vertcat(vec(F), q)
        lam = SX.sym('lam', self.n_lam)
        lcp_cstr = F @ lam + q
        lcp_loss = dot(lam, lcp_cstr)
        quadprog = {'x': lam, 'f': lcp_loss, 'g': lcp_cstr, 'p': para_vec}
        opts = {'printLevel': 'none', }
        self.lcpSolver = qpsol('lcpSolver', 'qpoases', quadprog, opts)

    def dyn_fn(self, x, u, lam):
        pos_cart = x[0]
        pos_pole = x[1]
        vel_cart = x[2]
        vel_pole = x[3]
        lam_1 = lam[0]
        lam_2 = lam[1]

        dot_pos_cart = vel_cart
        dot_pos_pole = vel_pole
        dot_vel_cart = 1 / (self.mc + self.mp * sin(np.pi - pos_pole) * sin(np.pi - pos_pole)) * (
                u + self.mp * sin(np.pi - pos_pole) * (
                self.len_p * vel_pole * vel_pole + self.g * cos(np.pi - pos_pole)))
        dot_vel_pole = 1 / (self.mc + self.mp * sin(np.pi - pos_pole) * sin(np.pi - pos_pole)) / self.len_p * (
                -u * cos(np.pi - pos_pole) - self.mp * self.len_p * vel_pole * vel_pole * sin(np.pi - pos_pole) * cos(
            np.pi - pos_pole) - (self.mp + self.mc) * self.g * sin(np.pi - pos_pole)
        ) + 1 / (self.len_p * self.mp) * lam_1 - 1 / (self.len_p * self.mp) * lam_2

        dot_x = np.array([dot_pos_cart, dot_pos_pole, dot_vel_cart.item(), dot_vel_pole.item()])
        x_next = x + dot_x * self.Ts

        return x_next.flatten()

    def wall_lcp(self, x, u):
        F = 1 / self.ks * np.eye(2)
        F = np.asarray(F)

        pos_cart = x[0]
        pos_pole = x[1]
        q1 = self.d1 - sin(pos_pole) * self.len_p - pos_cart
        q2 = -self.d2 + pos_cart + sin(pos_pole) * self.len_p

        q = np.array([q1, q2])

        return F, q

    def random_datagen(self, data_size, noise_level=0.0, magnitude_x=0.1, magnitude_u=1):
        x_batch = magnitude_x * np.random.uniform(-1, 1, size=(data_size, self.n_state))
        x_batch = x_batch + noise_level * np.random.randn(*x_batch.shape)
        u_batch = magnitude_u * np.random.uniform(-1, 1, size=(data_size, self.n_control))
        u_batch = u_batch + noise_level * np.random.randn(*u_batch.shape)

        # compute the lam vector
        para_batch = DM.zeros(self.n_lam * self.n_lam + self.n_lam, data_size)
        for i in range(data_size):
            x = x_batch[i]
            u = u_batch[i]
            F, q = self.wall_lcp(x, u)
            para_batch[:, i] = vertcat(vec(F), q)

        # solve LCP for lam
        sol = self.lcpSolver(p=para_batch, lbx=0., lbg=0.)
        lam_batch = sol['x'].full().T

        x_next_batch = np.zeros((data_size, self.n_state))
        for i in range(data_size):
            x = x_batch[i]
            u = u_batch[i]
            lam = lam_batch[i]
            x_next_batch[i] = self.dyn_fn(x, u, lam)

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

    def forward(self, x, u):
        F, q = self.wall_lcp(x, u)
        para = vertcat(vec(F), q)
        sol = self.lcpSolver(p=para, lbx=0., lbg=0.)
        lam = sol['x'].full().flatten()

        return self.dyn_fn(x, u, lam), lam


# do statistics for the modes
def statiModes(lam_batch, tol=1e-5):
    # dimension of the lambda
    n_lam = lam_batch.shape[1]
    # total number of modes
    total_n_mode = float(2 ** n_lam)

    # do the statistics for the modes
    lam_batch_mode = np.where(lam_batch < tol, 0, 1)
    unique_mode_list, mode_count_list = np.unique(lam_batch_mode, axis=0, return_counts=True)
    mode_frequency_list = mode_count_list / total_n_mode

    active_mode_frequence = unique_mode_list.shape[0] / total_n_mode
    # print(active_mode_frequence, total_n_mode)

    return active_mode_frequence, unique_mode_list, mode_frequency_list
