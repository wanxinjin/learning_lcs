import cartpole_class
import lcs.optim as opt
import numpy.linalg as la
from casadi import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def print(*args):
    __builtins__.print(*("%.5f" % a if isinstance(a, float) else a
                         for a in args))


# color list
color_list = np.linspace(0, 1, 10)





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

# ==============================   load the real testing data   ==================================
test_data = np.load('../data/test_data.npy', allow_pickle=True).item()
n_state = test_data['n_state']
n_control = test_data['n_control']
n_lam = test_data['n_lam']

test_x_batch = test_data['test_x']
test_u_batch = test_data['test_u']
test_x_next_batch = test_data['test_x_next']
test_lam_batch = test_data['test_lam']

test_data_size = test_x_batch.shape[0]

test_mode_list, test_mode_frequency_list = cartpole_class.statiModes(test_lam_batch)
print('number of modes in the testing data:', test_mode_frequency_list.size)
print('mode frequency in the testing data: ', test_mode_frequency_list)
# check the mode index
test_mode_list, test_mode_indices = cartpole_class.plotModes(test_lam_batch)


# ==============================   create the learner object    ========================================
learner = cartpole_class.cartpole_learner(n_state, n_control, n_lam=n_lam, stiffness=0.00)

# ================================   do some anlaysis for the prediction    ======================================
pred_x_next_batch, pred_lam_opt_batch = learner.dyn_prediction(test_x_batch, test_u_batch, learned_theta)
# compute the overall relative prediction error
error_x_next_batch = pred_x_next_batch - test_x_next_batch
relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(test_x_next_batch, axis=1)).mean()
# compute the predicted mode statistics
pred_mode_list0, pred_mode_frequency_list = cartpole_class.statiModes(pred_lam_opt_batch)
pred_mode_list1, pred_mode_indices = cartpole_class.plotModes(pred_lam_opt_batch)
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

if True:
    ini_theta=learned_res['theta_trace'][0]
    print('------------------------------------------------')
    print('A')
    print('initial:\n', learner.A_fn(ini_theta))
    print('learned:\n',learner.A_fn(learned_theta))
    print('------------------------------------------------')
    print('B')
    print('initial:\n', learner.B_fn(ini_theta))
    print('learned:\n',learner.B_fn(learned_theta))
    print('------------------------------------------------')
    print('C')
    print('initial:\n', learner.C_fn(ini_theta))
    print('learned: \n', learner.C_fn(learned_theta))
    print('------------------------------------------------')
    print('D')
    print('true')
    print('initial:')
    print(learner.D_fn(ini_theta))
    print('learned')
    print(learner.D_fn(learned_theta))
    print('------------------------------------------------')
    print('E')
    print('true')
    print('initial:')
    print(learner.E_fn(ini_theta))
    print('learned')
    print(learner.E_fn(learned_theta))
    print('------------------------------------------------')
    print('F')
    print('true')
    print('initial:')
    print(learner.F_fn(ini_theta))
    print('learned')
    print(learner.F_fn(learned_theta))
    print('------------------------------------------------')
    print('lcp_offset')
    print('initial:')
    print(learner.lcp_offset_fn(ini_theta))
    print('learned')
    print(learner.lcp_offset_fn(learned_theta))






print('------------------------------------------------')
error_x_next_batch = pred_x_next_batch - test_x_next_batch
relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(test_x_next_batch, axis=1)).mean()
print('relative prediction error:', relative_error)


print('------------------------------------------------')
print('pred_x/true_x')
print(pred_x_next_batch[0:10] / test_x_next_batch[0:10])
print('predicted x')
print(pred_x_next_batch[0:10])
print('true x')
print(test_x_next_batch[0:10])
print('max:', np.amax(pred_x_next_batch[0:10] / test_x_next_batch[0:10]))
print('min:', np.amin(pred_x_next_batch[0:10] / test_x_next_batch[0:10]))

print('------------------------------------------------')
np.set_printoptions(suppress=True)
print('pred_lam mode')
print(np.where(pred_lam_opt_batch[0:10] < 1e-5, 0, 1))
print('true lam mode')
print(np.where(test_lam_batch[0:10] < 1e-5, 0, 1))




plt.show()



# safe the learned matrix
np.save('learned_lcs',
        {
            'A': learner.A_fn(learned_theta).full(),
            'B': learner.B_fn(learned_theta).full(),
            'C': learner.C_fn(learned_theta).full(),
            'D': learner.D_fn(learned_theta).full(),
            'E': learner.E_fn(learned_theta).full(),
            'F': learner.F_fn(learned_theta).full(),
            'lcp_offset': learner.lcp_offset_fn(learned_theta).full(),
        }
        )
