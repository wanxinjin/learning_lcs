import time

from lcs import lcs_test
import numpy as np
from casadi import *
import lcs.optim as opt
from numpy import linalg as la
from lcs import lcs_control
import matplotlib.pyplot as plt

# load the system
lcs_mats = np.load('../cartpole_debug/random_lcs.npy', allow_pickle=True).item()
min_sig = lcs_mats['min_sig']
# set the learner parameter
n_theta = lcs_mats['n_theta']
n_state = lcs_mats['n_state']
n_control = lcs_mats['n_control']
n_lam = lcs_mats['n_lam']

# establish the VN learner (violation-based method)
qp_learner = lcs_test.QP_learner(n_state=n_state, n_control=n_control, n_lam=n_lam)
qp_optimizier = opt.Adam()
qp_optimizier.learning_rate = 1e-2

# generate testing data
testing_data_size = 1000
testing_data = lcs_test.gen_data(lcs_mats, testing_data_size,
                                 noise_level=0.0, magnitude_x=0.5, magnitude_u=2)
# testing the prediction error
x_test_batch = testing_data['x_batch']
u_test_batch = testing_data['u_batch']
x_next_test_batch = testing_data['x_next_batch']
lam_test_batch = testing_data['lam_batch']


# --------------------------- testing the prediction error for PN and VN ----------------------------------#
# load the learned
load_learned = np.load('learned_lcs_test.npy', allow_pickle=True).item()
qp_theta_trace = load_learned['qp_theta_trace']
qp_loss_trace = load_learned['qp_loss_trace']
qp_theta = qp_theta_trace[-1]



pred_lam_batch = qp_learner.qpSol(theta=qp_theta_trace[-1], x_batch=x_test_batch, u_batch=u_test_batch)
error_lam_batch = pred_lam_batch - lam_test_batch
relative_error = ((error_lam_batch * error_lam_batch).sum(axis=1).mean()) / \
                 ((lam_test_batch * lam_test_batch).sum(axis=1).mean())


print('------------------------------------------------')
print('relative prediction error:', relative_error)

print('------------------------------------------------')
print('D')
print(lcs_mats['D'])
print(qp_learner.D_fn(qp_theta))

print('------------------------------------------------')
print('E')
print(lcs_mats['E'])
print(qp_learner.E_fn(qp_theta))

print('------------------------------------------------')
print('F')
print(lcs_mats['F'])
print(qp_learner.F_fn(qp_theta))

print('------------------------------------------------')
print('F')
print(lcs_mats['lcp_offset'])
print(qp_learner.lcp_offset_fn(qp_theta))


print('------------------------------------------------')
x_seg_batch = x_test_batch[:6].T
u_seg_batch = u_test_batch[:6].T
lam_seg_batch = lam_test_batch[:6].T
x_next_seg_batch = x_next_test_batch[:6].T

pred_lam_seg_batch = pred_lam_batch[:6].T

print(lam_seg_batch)
print(pred_lam_seg_batch)

print('------------------------------------------------')
print(lam_test_batch[6:12].T)
print(pred_lam_batch[6:12].T)

breakpoint()
