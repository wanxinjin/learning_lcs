from lcs import lcs_control
import numpy as np
import matplotlib.pyplot as plt
import time

#
# load the system
true_sys = np.load('../cartpole_debug/cartpole_lcs.npy', allow_pickle=True).item()
n_state = true_sys['n_state']
n_control = true_sys['n_control']
n_lam = true_sys['n_lam']
true_A = true_sys['A']
true_B = true_sys['B']
true_C = true_sys['C']
true_dyn_offset = true_sys['dyn_offset']
true_D = true_sys['D']
true_E = true_sys['E']
true_F = true_sys['F']
true_lcp_offset = true_sys['lcp_offset']

# n_state = 8
# n_control = 2
# n_lam = 3
# gen_stiffness = 1
#
# lcs_mats = lcs_learning2.gen_lcs(n_state, n_control, n_lam, gen_stiffness)
# true_A = lcs_mats['A']
# true_B = lcs_mats['B']
# true_C = lcs_mats['C']
# true_dyn_offset = lcs_mats['dyn_offset']
# true_D = lcs_mats['D']
# true_E = lcs_mats['E']
# true_F = lcs_mats['F']
# true_lcp_offset = lcs_mats['lcp_offset']

true_sys = lcs_control.LCS_MPC(A=true_A,
                               B=true_B,
                               C=true_C,
                               dyn_offset=true_dyn_offset,
                               D=true_D,
                               E=true_E,
                               F=true_F,
                               lcp_offset=true_lcp_offset
                               )

gt_mpc = lcs_control.LCS_MPC(A=true_A,
                             B=true_B,
                             C=true_C,
                             dyn_offset=true_dyn_offset,
                             D=true_D,
                             E=true_E,
                             F=true_F,
                             lcp_offset=true_lcp_offset
                             )

# print the results
Q = 1 * np.eye(n_state)
R = 0.1 * np.eye(n_control)
QN = 20 * np.eye(n_state)
mpc_horizon = 50
init_state = np.random.randn(n_state)
sim_horizon = 50

st = time.time()
sol1 = gt_mpc.mpc(init_state, mpc_horizon, Q, R, QN)
print(time.time() - st)
st = time.time()
sol2 = gt_mpc.mpc_penalty(init_state, mpc_horizon, Q, R, QN, epsilon=1e-5)
print(time.time() - st)

# plot the results

plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.plot(sol1['state_traj_opt'][:, 0], label='true')
plt.plot(sol2['state_traj_opt'][:, 0], label='penalty')
plt.legend()
plt.subplot(1, 4, 2)
plt.plot(sol1['state_traj_opt'][:, 1], label='true')
plt.plot(sol2['state_traj_opt'][:, 1], label='penalty')
plt.legend()
plt.subplot(1, 4, 3)
plt.plot(sol1['state_traj_opt'][:, 2], label='true')
plt.plot(sol2['state_traj_opt'][:, 2], label='penalty')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(sol1['state_traj_opt'][:, 3], label='true')
plt.plot(sol2['state_traj_opt'][:, 3], label='penalty')
plt.legend()

plt.tight_layout()
plt.show()
