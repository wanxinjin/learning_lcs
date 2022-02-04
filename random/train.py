import test_class
import lcs.optim as opt
import numpy.linalg as la
from casadi import *
import matplotlib.pyplot as plt


def print(*args):
    __builtins__.print(*("%.5f" % a if isinstance(a, float) else a
                         for a in args))


# color list
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
color_list2 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:olive']
color_list3=np.array([0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.4])

# load the system
lcs_mats = np.load('random_lcs.npy', allow_pickle=True).item()
min_sig = lcs_mats['min_sig']

# generating the testing data and training data
n_state = lcs_mats['n_state']
n_lam = lcs_mats['n_lam']
A = lcs_mats['A']
C = lcs_mats['C']
D = lcs_mats['D']
G = lcs_mats['G']
F = lcs_mats['F']
lcp_offset = lcs_mats['lcp_offset']

# ====================================================================================================
# generate the training data
data_generator = test_class.LCS_learner(n_state, n_lam, A, C, D, G, lcp_offset, stiffness=0)
# ====================================================================================================
# generate the training data
train_data_size = 500
train_x_batch = 1 * np.random.uniform(-1, 1, size=(train_data_size, n_state))
train_x_next_batch, train_lam_opt_batch = data_generator.dyn_prediction(train_x_batch, theta_val=[])
mode_percentage, unique_mode_list, mode_frequency_list = test_class.statiModes(train_lam_opt_batch)

# check the mode index (this is for plotting)
mode_list, train_mode_indices = test_class.plotModes(train_lam_opt_batch)

# do the learning process
learner = test_class.LCS_learner(n_state, n_lam=n_lam, stiffness=10)
# print(learner.theta)
true_theta = vertcat(vec(G), vec(D), lcp_offset, vec(A), vec(C)).full().flatten()

# ====================================================================================================
# plot dimension index
x_indx = 0
y_indx = 0

plt.ion()
fig, ax = plt.subplots()
# plot the training data
train_x = train_x_batch[:, x_indx]
train_y = train_x_next_batch[:, y_indx]
for i in range(train_data_size):
    ax.scatter(train_x[i], train_y[i], c=color_list[train_mode_indices[i]])
# plot the learned
pred_x, pred_y = [], []
sc = ax.scatter(pred_x, pred_y, s=10)
plt.draw()

# ====================================================================================================
# doing learning process
curr_theta = 0.1 * np.random.randn(learner.n_theta)
# curr_theta = true_theta + 2 * np.random.randn(learner.n_theta)
mini_batch_size = 100
loss_trace = []
theta_trace = []
# optimizer
optimizier = opt.Adam()
optimizier.learning_rate = 1e-2
for k in range(5000):
    # mini batch dataset
    shuffle_index = np.random.permutation(train_data_size)[0:mini_batch_size]
    x_mini_batch = train_x_batch[shuffle_index]
    x_next_mini_batch = train_x_next_batch[shuffle_index]
    lam_mini_batch = train_lam_opt_batch[shuffle_index]

    # compute the lambda batch
    lam_phi_opt_mini_batch, loss_opt_batch = learner.compute_lambda(x_mini_batch, x_next_mini_batch, curr_theta)

    # compute the gradient
    dtheta, loss, dyn_loss, lcp_loss, dtheta_hessian = \
        learner.gradient_step(x_mini_batch, x_next_mini_batch, curr_theta, lam_phi_opt_mini_batch, second_order=False)

    # store and update
    loss_trace += [loss]
    theta_trace += [curr_theta]
    curr_theta = optimizier.step(curr_theta, dtheta)
    # curr_theta = optimizier.step(curr_theta, dtheta_hessian)

    if k % 100 == 0:
        # one the prediction to check the prediction mini_batch
        pred_x_next_batch, pred_lam_batch = learner.dyn_prediction(x_mini_batch, curr_theta)
        error_x_next_batch = pred_x_next_batch - x_next_mini_batch
        relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(x_next_mini_batch, axis=1)).mean()

        # do the plot
        pred_x_next_full, pred_lam_full = learner.dyn_prediction(train_x_batch, curr_theta)
        pred_x = train_x_batch[:, x_indx]
        pred_y = pred_x_next_full[:, y_indx]


        # check the mode index (this is for plotting)
        mode_list, pred_mode_indices = test_class.plotModes(pred_lam_full)



        # this is for plot
        sc.set_offsets(np.c_[pred_x, pred_y])
        sc.set_array(color_list3[pred_mode_indices])





        fig.canvas.draw_idle()
        plt.pause(0.1)

        print(
            '| iter', k,
            '| loss:', loss,
            '| grad:', norm_2(dtheta),
            '| dyn_loss:', dyn_loss,
            '| lcp_loss:', lcp_loss,
            '| relative_error:', relative_error,
        )

# print(theta_trace[-1])

# save
np.save('learned', {
    'theta_trace': theta_trace,
    'loss_trace': loss_trace,
})
