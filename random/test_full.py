import test_class
import numpy.linalg as la
from casadi import *


def print(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))


# load the system
lcs_mats = np.load('random_lcs.npy', allow_pickle=True).item()
min_sig = lcs_mats['min_sig']
print(min_sig)

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
# make sure this is consistent with  line 29-30 in  train_full.py
# generate the training data
data_generator = test_class.LCS_learner(n_state, n_lam, A, C, D, G, lcp_offset, stiffness=0)
# ====================================================================================================

# generate the training data
test_data_size = 5000
test_x_batch = 1 * np.random.uniform(-1, 1, size=(test_data_size, n_state))
test_x_next_batch, test_lam_opt_batch = data_generator.dyn_prediction(test_x_batch, theta_val=[])

# load the learned results
learned = np.load('learned.npy', allow_pickle=True).item()
learned_theta = learned['theta_trace'][-1]


# ====================================================================================================
learner = test_class.LCS_learner(n_state, n_lam=n_lam, stiffness=1)
# ====================================================================================================

pred_x_next_batch, pred_lam_opt_batch = learner.dyn_prediction(test_x_batch, learned_theta)


print('------------------------------------------------')
print('A')
print(A)
print(learner.A_fn(learned_theta))
print(A/learner.A_fn(learned_theta))
print('------------------------------------------------')
print('C')
print(C)
print(learner.C_fn(learned_theta))
print('------------------------------------------------')
print('D')
print(D)
print(learner.D_fn(learned_theta))
print('------------------------------------------------')
print('G')
print(G)
print(learner.G_fn(learned_theta))
print('------------------------------------------------')
print('lcp_offset')
print(lcp_offset)
print(learner.lcp_offset_fn(learned_theta))


print('------------------------------------------------')
error_x_next_batch = pred_x_next_batch - test_x_next_batch
relative_error = (la.norm(error_x_next_batch, axis=1) / la.norm(test_x_next_batch, axis=1)).mean()
print('relative_error:', relative_error)
print('------------------------------------------------')
print('pred_x/true_x')
print(pred_x_next_batch[0:10]/test_x_next_batch[0:10])
print('prediction')
print(pred_x_next_batch[0:10])
print('true')
print(test_x_next_batch[0:10])



print('max:', np.amax(pred_x_next_batch[0:10]/test_x_next_batch[0:10]))
print('min:', np.amin(pred_x_next_batch[0:10]/test_x_next_batch[0:10]))
print('------------------------------------------------')
np.set_printoptions(suppress=True)
print('pred_lam')
print(pred_lam_opt_batch[0:10])
print('test_lam')
print(test_lam_opt_batch[0:10])






