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

# ==============================   generate the training data    ========================================
# ！！！！！！！！！！！！！！！！make sure matching with line 27-33 in the train.py
data_generator = test_class.LCS_learner(n_state, n_lam, A, C, D, G, lcp_offset, stiffness=0)
# generate the testing data
test_data_size = 1000
test_x_batch = 1 * np.random.uniform(-1, 1, size=(test_data_size, n_state))
test_x_next_batch, test_lam_opt_batch = data_generator.dyn_prediction(test_x_batch, theta_val=[])
# check the mode index
test_mode_list, test_mode_indices = test_class.plotModes(test_lam_opt_batch)

# ==============================   create the learner object    ========================================
# ！！！！！！！！！！！！！！！！ make sure matching with line 60-63 in the train.py
learner = test_class.LCS_learner(n_state, n_lam=n_lam, stiffness=10)
# load learning results
learned_res = np.load('learned.npy', allow_pickle=True).item()
learned_theta = learned_res['theta_trace'][-1]
pred_x_next_batch, pred_lam_opt_batch = learner.dyn_prediction(test_x_batch, learned_theta)
pred_mode_list, pred_mode_indices = test_class.plotModes(test_lam_opt_batch)

# =============== plot the training data, each color for each mode  ======================================
x_indx = 0
y_indx = 0
# plot the true one
fig, ax = plt.subplots()
ax.set_title('Learned modes marked in (+), \n True modes marked in (o)')
test_x = test_x_batch[:, x_indx]
test_y = test_x_next_batch[:, y_indx]
plt.scatter(test_x, test_y, c=color_list[test_mode_indices], s=80, alpha=0.3)
# plot the predicted one
pred_y = pred_x_next_batch[:, y_indx]
ax.scatter(test_x, pred_y, c=color_list[pred_mode_indices], s=30, marker="+")
# plt.show()

# ==================== print some key results  =======================

print('------------------------------------------------')
error_x_next_batch = pred_x_next_batch - test_x_next_batch
relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(test_x_next_batch, axis=1)).mean()
print('relative prediction error:', relative_error)

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
print(pred_lam_opt_batch[0:10])
print('test_lam')
print(test_lam_opt_batch[0:10])

print('------------------------------------------------')
print('compute the distance')
pred_dist = learner.dist_fn(test_x_batch.T, pred_lam_opt_batch.T, learned_theta).full().T
print(pred_dist[0:10])
print('prod  the distance with lam')
print(pred_dist[0:10]*pred_lam_opt_batch[0:10])
