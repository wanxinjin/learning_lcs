import time

from lcs import lcs_test
import numpy as np
from casadi import *
import lcs.optim as opt
from lcs import lcs_control
import matplotlib.pyplot as plt
import numpy.linalg as la

# load the system
lcs_mats = np.load('../cartpole_debug/random_lcs.npy', allow_pickle=True).item()
min_sig = lcs_mats['min_sig']

# generating the testing data and training data
training_data_size = 1000
training_data = lcs_test.gen_data(lcs_mats, training_data_size, noise_level=0,
                                  magnitude_x=1.5, magnitude_u=5)
print('mode_percentage', training_data['mode_percentage'])

# generate testing data
testing_data_size = 300
testing_data = lcs_test.gen_data(lcs_mats, testing_data_size, noise_level=0.0, magnitude_x=1.5, magnitude_u=5)

# set the learner parameter
n_theta = lcs_mats['n_theta']
n_state = lcs_mats['n_state']
n_control = lcs_mats['n_control']
n_lam = lcs_mats['n_lam']

# establish the VN learner (violation-based method)
qp_learner = lcs_test.QP_learner(n_state=n_state, n_control=n_control, n_lam=n_lam)
qp_optimizier = opt.Adam()
qp_optimizier.learning_rate = 1e-2


# ------------------------------------------ training process ----------------------------------------
# initialization
qp_curr_theta = 0.01 * np.random.randn(qp_learner.n_theta)

# storage
qp_loss_trace = []
qp_theta_trace = []

# load data
x_batch = training_data['x_batch']
u_batch = training_data['u_batch']
x_next_batch = training_data['x_next_batch']
goal_lam_batch = training_data['lam_batch']

# learning process
max_iter = 10000
mini_batch_size = 300
for iter in range(max_iter):
    # mini_batch_size
    shuffle_index = np.random.permutation(training_data_size)[0:mini_batch_size]
    x_mini_batch = x_batch[shuffle_index]
    u_mini_batch = u_batch[shuffle_index]
    x_next_mini_batch = x_next_batch[shuffle_index]
    goal_lam_mini_batch = goal_lam_batch[shuffle_index]

    # solve lam
    curr_lam_mini_sol = qp_learner.qpSol(qp_curr_theta, x_mini_batch, u_mini_batch)

    # gradient
    loss, dtheta, res = qp_learner.gradient(qp_curr_theta, x_mini_batch, u_mini_batch, curr_lam_mini_sol,
                                            goal_lam_mini_batch)

    qp_loss_trace += [loss]
    qp_theta_trace += [qp_curr_theta]
    qp_curr_theta = qp_optimizier.step(qp_curr_theta, dtheta)

    if iter % 100 == 0:
        print(iter, 'loss:', loss, 'grad norm:', norm_2(dtheta), 'kkt norm:', res)

# --------------------------- testing the prediction error for PN and VN ----------------------------------#
# testing the prediction error
x_test_batch = testing_data['x_batch']
u_test_batch = testing_data['u_batch']
x_next_test_batch = testing_data['x_next_batch']
lam_test_batch = testing_data['lam_batch']

pred_lam_batch = qp_learner.qpSol(theta=qp_theta_trace[-1], x_batch=x_test_batch, u_batch=u_test_batch)
error_lam_batch = pred_lam_batch - lam_test_batch
relative_error = ((error_lam_batch * error_lam_batch).sum(axis=1).mean()) / \
                 ((lam_test_batch * lam_test_batch).sum(axis=1).mean())

print('------------------------------------------------')
print('Prediction error for VN:', relative_error)

# save
np.save('learned_lcs_test', {
    'qp_theta_trace': qp_theta_trace,
    'qp_loss_trace': qp_loss_trace,
    'qp_pred_error': relative_error,
    'n_state': n_state,
    'n_control': n_control,
    'n_lam': n_lam,
})
