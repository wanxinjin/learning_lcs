import test_class
import lcs.optim as opt
import numpy.linalg as la
from casadi import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def print(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))


# color list
color_list = np.linspace(0, 1, 10)

# ==============================   load the generated LCS system   ==================================
lcs_mats = np.load('random_lcs.npy', allow_pickle=True).item()
n_state = lcs_mats['n_state']
n_lam = lcs_mats['n_lam']
A = lcs_mats['A']
C = lcs_mats['C']
D = lcs_mats['D']
G = lcs_mats['G']
F = lcs_mats['F']
lcp_offset = lcs_mats['lcp_offset']


# ==============================# re-show the training plot   ==================================
learned_res = np.load('learned.npy', allow_pickle=True).item()
learned_theta = learned_res['theta_trace'][-1]
print('####################training results analysis#########################')
print('mode count for training data:', learned_res['train_mode_count'])
print('mode list for training data:\n', learned_res['train_mode_list'])
print('mode frequency for training data:\n', learned_res['train_mode_frequency'])
print('mode count for prediction data:', learned_res['pred_mode_count'])
print('mode list for prediction data:\n', learned_res['pred_mode_list'])
print('mode frequency for prediction data:\n', learned_res['pred_mode_frequency'])
print('prediction accuracy for each mode:\n', learned_res['pred_error_per_mode_list'])
print('overall relative prediction error: \n', learned_res['relative_error'])
print('---------------------------------------------------------------------------')


# plot the true one
fig0, ax = plt.subplots()
train_x = learned_res['train_x']
train_y = learned_res['train_y']
train_mode_indices = learned_res['train_mode_indices']
learned_y = learned_res['pred_y']
learned_mode_indices = learned_res['pred_mode_indices']
ax.set_title('Learned modes marked in (^), \n True modes marked in (o)')
plt.scatter(train_x, train_y, c=color_list[train_mode_indices], s=80, alpha=0.3)
# plot the predicted one
ax.scatter(train_x, learned_y, c=color_list[learned_mode_indices], s=40, marker="^")
plt.show()




# ==============================   generate the testing data    ========================================
# ！！！！！！！！！！！！！！！！make sure matching with line 27-33 in the train.py
data_generator = test_class.LCS_learner(n_state, n_lam, A, C, D, G, lcp_offset, stiffness=0)
# generate the testing data
test_data_size = 1000
test_x_batch = 1 * np.random.uniform(-1, -0.5, size=(test_data_size, n_state))
test_x_next_batch, test_lam_opt_batch = data_generator.dyn_prediction(test_x_batch, theta_val=[])
# check the mode index
test_mode_list, test_mode_frequency_list = test_class.statiModes(test_lam_opt_batch)
test_mode_list, test_mode_indices = test_class.plotModes(test_lam_opt_batch)

# ==============================   create the learner object    ========================================
# ！！！！！！！！！！！！！！！！ make sure matching with line 60-63 in the train.py
learner = test_class.LCS_learner(n_state, n_lam=2, stiffness=10)



# ================================   do some anlaysis for the prediction    ======================================
pred_x_next_batch, pred_lam_opt_batch = learner.dyn_prediction(test_x_batch, learned_theta)
# compute the overall relative prediction error
error_x_next_batch = pred_x_next_batch - test_x_next_batch
relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(test_x_next_batch, axis=1)).mean()
# compute the predicted mode statistics
pred_mode_list0, pred_mode_frequency_list = test_class.statiModes(pred_lam_opt_batch)
pred_mode_list1, pred_mode_indices = test_class.plotModes(pred_lam_opt_batch)
pred_error_per_mode_list = []
for i in range(len(pred_mode_list0)):
    mode_i_index = np.where(pred_mode_indices == i)
    mode_i_error = error_x_next_batch[mode_i_index]
    mode_i_relative_error = (la.norm(mode_i_error, axis=1) / la.norm(test_x_next_batch[mode_i_index], axis=1)).mean()
    pred_error_per_mode_list += [mode_i_relative_error]

print('#################### testing results analysis#########################')
print('mode count for testing data:', test_mode_frequency_list.size)
print('mode list for testing data:\n', test_mode_list)
print('mode frequency for testing data:\n', test_mode_frequency_list)
print('mode count for prediction data:', pred_mode_frequency_list.size)
print('mode list for prediction data:\n', pred_mode_list0)
print('mode frequency for prediction data:\n', pred_mode_frequency_list)
print('prediction accuracy for each mode:\n', pred_error_per_mode_list)
print('---------------------------------------------------------------------------')



# =============== plot the training data, each color for each mode  ======================================
plot_x_indx = 0
plot_y_indx = 0


# plot the true one
fig1, ax = plt.subplots()
ax.set_title('True modes marked in (o)')
test_x = test_x_batch[:, plot_x_indx]
test_y = test_x_next_batch[:, plot_y_indx]
plt.scatter(test_x, test_y, c=color_list[test_mode_indices], s=80, alpha=1)


# plot the true one
fig2, ax = plt.subplots()
ax.set_title('Learned modes marked in (^), \n True modes marked in (o)')
test_x = test_x_batch[:, plot_x_indx]
test_y = test_x_next_batch[:, plot_y_indx]
plt.scatter(test_x, test_y, c=color_list[test_mode_indices], s=80, alpha=0.3)
# plot the predicted one
pred_y = pred_x_next_batch[:, plot_y_indx]
ax.scatter(test_x, pred_y, c=color_list[pred_mode_indices], s=40, marker="^")
# plt.show()

# plot the true one
fig3, ax = plt.subplots()
ax.set_title('Learned modes marked in (^)')
test_x = test_x_batch[:, plot_x_indx]
test_y = test_x_next_batch[:, plot_y_indx]
plt.scatter(test_x, test_y, c=color_list[test_mode_indices], s=0, alpha=0.3)
# plot the predicted one
pred_y = pred_x_next_batch[:, plot_y_indx]
ax.scatter(test_x, pred_y, c=color_list[pred_mode_indices], s=40, marker="^")
# plt.show()


# ==================== print some key results  =======================

print('------------------------------------------------')
error_x_next_batch = pred_x_next_batch - test_x_next_batch
relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(test_x_next_batch, axis=1)).mean()
print('relative prediction error:', relative_error)

if False:
    print('------------------------------------------------')
    print('A')
    print('true')
    print(A)
    print('learned')
    print(learner.A_fn(learned_theta))
    print('------------------------------------------------')
    print('C')
    print('true')
    print(C)
    print('learned')
    print(learner.C_fn(learned_theta))
    print('------------------------------------------------')
    print('D')
    print('true')
    print(D)
    print(learner.D_fn(learned_theta))
    print('------------------------------------------------')
    print('G')
    print('true')
    print(G)
    print('learned')
    print(learner.G_fn(learned_theta))
    print('------------------------------------------------')
    print('lcp_offset')
    print(lcp_offset)
    print('learned')
    print(learner.lcp_offset_fn(learned_theta))

print('------------------------------------------------')
print('pred_x/true_x')
print(pred_x_next_batch[0:10] / test_x_next_batch[0:10])
print('prediction')
print(pred_x_next_batch[0:10])
print('true')
print(test_x_next_batch[0:10])
print('max:', np.amax(pred_x_next_batch[0:10] / test_x_next_batch[0:10]))
print('min:', np.amin(pred_x_next_batch[0:10] / test_x_next_batch[0:10]))

print('------------------------------------------------')
np.set_printoptions(suppress=True)
print('pred_lam')
print(np.where(pred_lam_opt_batch[0:10] < 1e-5, 0, 1))
print('test_lam')
print(np.where(test_lam_opt_batch[0:10] < 1e-5, 0, 1))

# # print('------------------------------------------------')
# print('compute the distance')
# pred_dist, pred_lcp_loss = learner.lcp_fn(test_x_batch.T, pred_lam_opt_batch.T, learned_theta)
# pred_dist = pred_dist.full().T
# pred_lcp_loss = pred_lcp_loss.full().T
# print(pred_dist[0:10])
# print('prod  the distance with lam')
# print(pred_dist[0:10] * pred_lam_opt_batch[0:10])


plt.show()