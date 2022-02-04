import time

from lcs import lcs_test
import numpy as np
from casadi import *
import lcs.optim as opt
from numpy import linalg as la
from lcs import lcs_control
import matplotlib.pyplot as plt

# load the system
lcs_mats = np.load('random_lcs.npy', allow_pickle=True).item()
min_sig = lcs_mats['min_sig']
# set the learner parameter
n_theta = lcs_mats['n_theta']
n_state = lcs_mats['n_state']
n_control = lcs_mats['n_control']
n_lam = lcs_mats['n_lam']

# generate testing data
testing_data_size = 1000
testing_data = lcs_test.gen_data(lcs_mats, testing_data_size,
                                 noise_level=0.0, magnitude_x=1.0, magnitude_u=10)

# --------------------------- testing the prediction error for PN and VN ----------------------------------#
# load the learned
load_learned = np.load('learned_lcs5.npy', allow_pickle=True).item()
vn_theta_trace = load_learned['vn_theta_trace']
vn_loss_trace = load_learned['vn_loss_trace']
vn_theta = vn_theta_trace[-1]

# =============================================================================================
given_lcp_offset = np.array([0.35, 0.35])
F_stiffness =1.0


# establish the VN learner (violation-based method)
gamma = 1e-3
epsilon = 1e0
vn_learning_rate = 1e-1
vn_learner = lcs_test.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam,
                             A=lcs_mats['A'],
                             B=lcs_mats['B'],
                             dyn_offset=lcs_mats['dyn_offset'],
                             C=lcs_mats['C'],
                             # D=lcs_mats['D'],
                             # E=lcs_mats['E'],
                             # G=lcs_mats['G'],
                             # H=lcs_mats['H'],
                             # lcp_offset=lcs_mats['lcp_offset'],
                             # F_stiffness=F_stiffness
                             )
vn_learner.diff2(gamma=gamma, epsilon=epsilon)
vn_optimizier = opt.Adam()
vn_optimizier.learning_rate = vn_learning_rate

# establish the testing object
tester = lcs_test.LCS_PN(n_state=n_state, n_control=n_control, n_lam=n_lam,
                         A=lcs_mats['A'],
                         B=lcs_mats['B'],
                         dyn_offset=lcs_mats['dyn_offset'],
                         C=lcs_mats['C'],
                         # D=lcs_mats['D'],
                         # E=lcs_mats['E'],
                         # G=lcs_mats['G'],
                         # H=lcs_mats['H'],
                         # lcp_offset=lcs_mats['lcp_offset'],
                         # F_stiffness=F_stiffness
                         )
tester.diff(w_C=0, C_ref=0, w_F=0, F_ref=0)
# =============================================================================================
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
print('dyn_offset')
print(lcs_mats['dyn_offset'])
print(vn_learner.dyn_offset_fn(vn_theta))

print('------------------------------------------------')
print('D')
print(lcs_mats['D'])
print(vn_learner.D_fn(vn_theta))

print('------------------------------------------------')
print('E')
print(lcs_mats['E'])
print(vn_learner.E_fn(vn_theta))


print('------------------------------------------------')
print('lcp_offset')
print(lcs_mats['lcp_offset'])
print(vn_learner.lcp_offset_fn(vn_theta))

print('------------------------------------------------')
print('F')
print(lcs_mats['F'])
print(vn_learner.F_fn(vn_theta))
print(min(np.linalg.eigvals(lcs_mats['F'] + lcs_mats['F'].T)))
print(la.eigvals(vn_learner.F_fn(vn_theta)+vn_learner.F_fn(vn_theta).T))


print('------------------------------------------------')
x_seg_batch = x_test_batch[:6].T
u_seg_batch = u_test_batch[:6].T
lam_seg_batch = lam_test_batch[:6].T
x_next_seg_batch = x_next_test_batch[:6].T

pred_next_seg_batch = pred_next_batch[:6].T
pred_lam_seg_batch = pred_lam_batch[:6].T

print('------------------------------------------------')
print('predict state')
print(pred_next_seg_batch)
print('true next state')
print(x_next_seg_batch)
print(pred_next_seg_batch/x_next_seg_batch)

print('------------------------------------------------')
print('predict lam')
print(pred_lam_seg_batch)
print('true next lam')
print(lam_seg_batch)
