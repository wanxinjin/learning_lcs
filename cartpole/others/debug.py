import time

from lcs import lcs_learning
import numpy as np
from casadi import *
import lcs.optim as opt
from numpy import linalg as la
from lcs import lcs_control
import matplotlib.pyplot as plt

# load the system
lcs_mats = np.load('../cartpole_debug/cartpole_lcs.npy', allow_pickle=True).item()
min_sig = lcs_mats['min_sig']
# set the learner parameter
F_stiffness = 1.0
n_theta = lcs_mats['n_theta']
n_state = lcs_mats['n_state']
n_control = lcs_mats['n_control']
n_lam = lcs_mats['n_lam']

# establish the VN learner (violation-based method)
gamma = 1e-3
epsilon = 1e1
vn_learner = lcs_learning.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam,
                                 dyn_offset=np.zeros(n_state),
                                 H=np.zeros((n_lam, n_lam)),
                                 F_stiffness=F_stiffness)
vn_learner.diff(gamma=gamma, epsilon=epsilon, w_C=1e-6, C_ref=0, w_F=0e-6, F_ref=0)

# generate testing data
testing_data_size = 1000
testing_data = lcs_learning.gen_data(lcs_mats, testing_data_size,
                                     noise_level=0.0, magnitude_x=0.5, magnitude_u=10)

# --------------------------- testing the prediction error for PN and VN ----------------------------------#
# load the learned
load_learned = np.load('../cartpole_debug/learned_lcs5.npy', allow_pickle=True).item()
vn_theta_trace = load_learned['vn_theta_trace']
vn_loss_trace = load_learned['vn_loss_trace']
vn_theta = vn_theta_trace[-1]

# establish the testing object
tester = lcs_learning.LCS_PN(n_state=n_state, n_control=n_control, n_lam=n_lam,
                             dyn_offset=np.zeros(n_state),
                             H=np.zeros((n_lam, n_lam)),
                             F_stiffness=F_stiffness)
tester.diff(w_C=0, C_ref=0, w_F=0, F_ref=0)

# testing the prediction error
x_test_batch = testing_data['x_batch']
u_test_batch = testing_data['u_batch']
x_next_test_batch = testing_data['x_next_batch']
lam_test_batch = testing_data['lam_batch']

vn_pred_error = tester.pred_error(batch_x=x_test_batch, batch_u=u_test_batch,
                                  batch_x_next=x_next_test_batch,
                                  current_theta=vn_theta)

pred_next_batch, pred_lam_batch = tester.pred(batch_x=x_test_batch, batch_u=u_test_batch, current_theta=vn_theta)

print('------------------------------------------------')
print('relative prediction error:', load_learned['vn_pred_error'])
print('final training loss:', vn_loss_trace[-1])

print('------------------------------------------------')
print('A')
print(lcs_mats['A'])
print(vn_learner.A_fn(vn_theta))
print('------------------------------------------------')
print('B')
print(lcs_mats['B'])
print(vn_learner.B_fn(vn_theta))
print('------------------------------------------------')
print('C')
print(lcs_mats['C'])
print(vn_learner.C_fn(vn_theta))


print('------------------------------------------------')
print('D')
print(lcs_mats['D'])
print(vn_learner.D_fn(vn_theta))



print('------------------------------------------------')
print('E')
print(lcs_mats['E'])
print(vn_learner.E_fn(vn_theta))



print('------------------------------------------------')
print('dyn_offset')
print(lcs_mats['dyn_offset'])
print(vn_learner.dyn_offset_fn(vn_theta))

print('------------------------------------------------')
print('lcp_offset')
print(lcs_mats['lcp_offset'])
print(vn_learner.lcp_offset_fn(vn_theta))


print('------------------------------------------------')
print('F')
print(lcs_mats['F'])
print(vn_learner.F_fn(vn_theta))



# print('------------------------------------------------')
# print(pred_next_batch[0:6].T)
# print(x_next_test_batch[0:6].T)
# relative_error = la.norm(pred_next_batch[:, :3] - x_next_test_batch[:, :3], axis=1) / la.norm(x_next_test_batch[:, :3],
#                                                                                               axis=1)
# print(relative_error.mean() * relative_error.mean())
# print('relative vn_pred_error:', vn_pred_error)

print('------------------------------------------------')
x_seg_batch = x_test_batch[:6].T
u_seg_batch = u_test_batch[:6].T
lam_seg_batch = lam_test_batch[:6].T
x_next_seg_batch = x_next_test_batch[:6].T

pred_next_seg_batch = pred_next_batch[:6].T
pred_lam_seg_batch = pred_lam_batch[:6].T


print(pred_next_seg_batch)
print(x_next_seg_batch)

print(pred_lam_seg_batch)
print(lam_seg_batch)


# print(
#     lcs_mats['A'] @ x_seg_batch + lcs_mats['B'] @ u_seg_batch + lcs_mats['C'] @ lam_seg_batch + lcs_mats['dyn_offset'])
# print(lcs_mats['C'] @ lam_seg_batch )
#
# print('------------------------------------------------')
# print(vn_learner.A_fn(vn_theta) @ x_seg_batch + vn_learner.B_fn(vn_theta) @ u_seg_batch + vn_learner.C_fn(
#     vn_theta) @ pred_lam_seg_batch + vn_learner.dyn_offset_fn(vn_theta))
# print(vn_learner.C_fn(
#     vn_theta) @ pred_lam_seg_batch + vn_learner.dyn_offset_fn(vn_theta))
#
# print('------------------------------------------------')
# print(lam_seg_batch)
# print(pred_lam_seg_batch)

