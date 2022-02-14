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
print('---------------------------------------------------------------------------')

# plot the true one
fig0, ax = plt.subplots()
ax.plot(learned_res['train_lam_batch'], label='true', linestyle='--')
ax.plot(learned_res['pred_lam_batch'], label='learned', linestyle='--')
ax.legend()
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
# ！！！！！！！！！！！！！！！！ make sure matching with line 60-63 in the train.py
learner = cartpole_class.cartpole_learner2(n_state, n_control, n_lam=n_lam,
                                           stiffness=1.)


# ================================   do some anlaysis for the prediction    ======================================
pred_lam_opt_batch = learner.dyn_prediction(test_x_batch, test_u_batch, learned_theta)

# plot the true one
fig1, ax = plt.subplots()
ax.plot(test_lam_batch, label='true', linestyle='--')
ax.plot(pred_lam_opt_batch, label='learned', linestyle='-')
ax.legend()
plt.show()




# ==================== print some key results  =======================

if True:
    ini_theta=learned_res['theta_trace'][0]
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








plt.show()
