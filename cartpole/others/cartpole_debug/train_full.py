import time

from lcs import lcs_test
import numpy as np
from casadi import *
import lcs.optim as opt
from lcs import lcs_control
import matplotlib.pyplot as plt

# load the system
lcs_mats = np.load('cartpole_lcs.npy', allow_pickle=True).item()
min_sig = lcs_mats['min_sig']

# generating the testing data and training data
training_data_size = 10000
training_data_noise_level = 0e-6
training_data = lcs_test.gen_data(lcs_mats, training_data_size, noise_level=training_data_noise_level,
                                  magnitude_x=1.0, magnitude_u=10)

# generate testing data
testing_data_size = 1000
testing_data = lcs_test.gen_data(lcs_mats, testing_data_size, noise_level=0.0, magnitude_x=1.0, magnitude_u=10)

# set the learner parameter
n_state = lcs_mats['n_state']
n_control = lcs_mats['n_control']
n_lam = lcs_mats['n_lam']

# =============================================================================================
given_lcp_offset = np.array([0.35, 0.35])
F_stiffness =1.0


# establish the VN learner (violation-based method)
gamma = 1e-3
epsilon = 1e1
vn_learning_rate = 1e-2
vn_learner = lcs_test.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam,
                             # A=lcs_mats['A'],
                             # B=lcs_mats['B'],
                             # dyn_offset=lcs_mats['dyn_offset'],
                             C=lcs_mats['C'],
                             # D=lcs_mats['D'],
                             # E=lcs_mats['E'],
                             # G=lcs_mats['G'],
                             # H=lcs_mats['H'],
                             # lcp_offset=lcs_mats['lcp_offset'],
                             # F_stiffness=F_stiffness
                             )
vn_learner.diff(gamma=gamma, epsilon=epsilon)
vn_optimizier = opt.Adam()
vn_optimizier.learning_rate = vn_learning_rate

# establish the testing object
tester = lcs_test.LCS_PN(n_state=n_state, n_control=n_control, n_lam=n_lam,
                         # A=lcs_mats['A'],
                         # B=lcs_mats['B'],
                         # dyn_offset=lcs_mats['dyn_offset'],
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


# initialization
init_magnitude = 0.1


vn_curr_theta = init_magnitude * np.random.randn(vn_learner.n_theta)



# storage
loss_trace = []
theta_trace = []

# load data
x_batch = training_data['x_batch']
u_batch = training_data['u_batch']
x_next_batch = training_data['x_next_batch']
lam_batch = training_data['lam_batch']

# learning process
max_iter = 5000
mini_batch_size = 500
for iter in range(max_iter):
    # mini_batch_size
    shuffle_index = np.random.permutation(training_data_size)[0:mini_batch_size]
    x_mini_batch = x_batch[shuffle_index]
    u_mini_batch = u_batch[shuffle_index]
    x_next_mini_batch = x_next_batch[shuffle_index]
    lam_mini_batch = lam_batch[shuffle_index]

    # compute the value for the lam_batch
    opt_lam_phi_batch, opt_mu_batch, opt_loss_batch, opt_lam_batch, opt_phi_batch = \
        vn_learner.compute_lam(batch_x=x_mini_batch,
                               batch_u=u_mini_batch,
                               batch_x_next=x_next_mini_batch,
                               curr_theta=vn_curr_theta)

    # compute the gradient
    vn_loss, dtheta, dyn_loss, lcp_loss = vn_learner.gradient(batch_x=x_mini_batch,
                                                              batch_u=u_mini_batch,
                                                              batch_x_next=x_next_mini_batch,
                                                              batch_lam_phi=opt_lam_phi_batch,
                                                              batch_mu=opt_mu_batch,
                                                              curr_theta=vn_curr_theta)

    # store
    loss_trace += [vn_loss]
    theta_trace += [vn_curr_theta]
    vn_curr_theta = vn_optimizier.step(vn_curr_theta, dtheta)

    if iter % 100 == 0:
        print(
            '| iter', iter,
            '| loss:', vn_loss,
            '| grad:', norm_2(dtheta),
            '| dyn_loss:', dyn_loss,
            '| lcp_loss:', lcp_loss,
        )

# --------------------------- testing the prediction error for PN and VN ----------------------------------#
# testing the prediction error
x_test_batch = testing_data['x_batch']
u_test_batch = testing_data['u_batch']
x_next_test_batch = testing_data['x_next_batch']

pred_error = tester.pred_error(batch_x=x_test_batch, batch_u=u_test_batch,
                               batch_x_next=x_next_test_batch,
                               current_theta=theta_trace[-1])

print('------------------------------------------------')
print('Prediction error:', pred_error)

# save
np.save('learned_lcs5', {
    'vn_theta_trace': theta_trace,
    'vn_loss_trace': loss_trace,
    'vn_pred_error': pred_error,
    'n_state': n_state,
    'n_control': n_control,
    'n_lam': n_lam,
    'F_stiffness': F_stiffness,
    'gamma': gamma,
    'epsilon': epsilon,
})
