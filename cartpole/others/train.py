import time

from lcs import lcs_learning
import numpy as np
from casadi import *
import lcs.optim as opt
from lcs import lcs_control
import matplotlib.pyplot as plt

# load the system
lcs_mats = np.load('../cartpole_debug/cartpole_lcs.npy', allow_pickle=True).item()
min_sig = lcs_mats['min_sig']

# generating the testing data and training data
training_data_size = 10000
training_data_noise_level = 0e-6
training_data = lcs_learning.gen_data(lcs_mats, training_data_size, noise_level=training_data_noise_level,
                                      magnitude_x=1.5, magnitude_u=10)
print('mode_percentage', training_data['mode_percentage'])

# generate testing data
testing_data_size = 1000
testing_data = lcs_learning.gen_data(lcs_mats, testing_data_size, noise_level=0.0, magnitude_x=1.5, magnitude_u=10)

# set the learner parameter
F_stiffness = 10
n_theta = lcs_mats['n_theta']
n_state = lcs_mats['n_state']
n_control = lcs_mats['n_control']
n_lam = lcs_mats['n_lam']

# establish the VN learner (violation-based method)
gamma = 1e-3
epsilon = 1e-3
vn_learning_rate = 5e-3
vn_learner = lcs_learning.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam,
                                 dyn_offset=np.zeros(n_state),
                                 H=np.zeros((n_lam,n_lam)),
                                 F_stiffness=F_stiffness, )
vn_learner.diff(gamma=gamma, epsilon=epsilon, w_C=1e-6, C_ref=0, w_F=0e-6, F_ref=0)
vn_optimizier = opt.Adam()
vn_optimizier.learning_rate = vn_learning_rate

# establish the testing object
tester = lcs_learning.LCS_PN(n_state=n_state, n_control=n_control, n_lam=n_lam,
                             dyn_offset=np.zeros(n_state),
                             H=np.zeros((n_lam, n_lam)),

                             F_stiffness=F_stiffness)
tester.diff(w_C=0, C_ref=0, w_F=0, F_ref=0)

# ------------------------------------------ training process ----------------------------------------
# initialization
init_magnitude = 0.1
vn_curr_theta = init_magnitude * np.random.randn(vn_learner.n_theta)

# storage
vn_loss_trace = []
vn_theta_trace = []

# load data
x_batch = training_data['x_batch']
u_batch = training_data['u_batch']
x_next_batch = training_data['x_next_batch']

# learning process
max_iter = 5000
mini_batch_size = 300
for iter in range(max_iter):
    # mini_batch_size
    shuffle_index = np.random.permutation(training_data_size)[0:mini_batch_size]
    x_mini_batch = x_batch[shuffle_index]
    u_mini_batch = u_batch[shuffle_index]
    x_next_mini_batch = x_next_batch[shuffle_index]

    # do one step for VN
    vn_mean_loss, vn_dtheta, vn_dyn_loss, vn_lcp_loss, relative_error = vn_learner.step(batch_x=x_mini_batch,
                                                                                        batch_u=u_mini_batch,
                                                                                        batch_x_next=x_next_mini_batch,
                                                                                        current_theta=vn_curr_theta)
    # store
    vn_loss_trace += [vn_mean_loss]
    vn_theta_trace += [vn_curr_theta]
    vn_curr_theta = vn_optimizier.step(vn_curr_theta, vn_dtheta)

    if iter % 100 == 0:
        print(
            '|| iter', iter,
            '|| VN_loss:', vn_mean_loss,
            '| VN_dyn_loss:', vn_dyn_loss,
            '| VN_lcp_loss:', vn_lcp_loss,
            '| relative_error:', relative_error,
        )

# --------------------------- testing the prediction error for PN and VN ----------------------------------#
# testing the prediction error
x_test_batch = testing_data['x_batch']
u_test_batch = testing_data['u_batch']
x_next_test_batch = testing_data['x_next_batch']

vn_pred_error = tester.pred_error(batch_x=x_test_batch, batch_u=u_test_batch,
                                  batch_x_next=x_next_test_batch,
                                  current_theta=vn_theta_trace[-1])

print('------------------------------------------------')
print('Prediction error for VN:', vn_pred_error)

# save
np.save('learned_lcs5', {
    'vn_theta_trace': vn_theta_trace,
    'vn_loss_trace': vn_loss_trace,
    'vn_pred_error': vn_pred_error,
    'n_state': n_state,
    'n_control': n_control,
    'n_lam': n_lam,
    'F_stiffness': F_stiffness,
    'gamma': gamma,
    'epsilon': epsilon,
})
