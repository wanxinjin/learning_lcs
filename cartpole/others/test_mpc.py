from lcs import lcs_learning
from lcs import lcs_control
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import numpy.linalg as la

# load the system
true_sys = np.load('../cartpole_debug/cartpole_lcs.npy', allow_pickle=True).item()
true_A = true_sys['A']
true_B = true_sys['B']
true_C = true_sys['C']
true_dyn_offset = true_sys['dyn_offset']
true_D = true_sys['D']
true_E = true_sys['E']
true_F = true_sys['F']
true_lcp_offset = true_sys['lcp_offset']

true_mpc = lcs_control.LCS_MPC(A=true_A,
                               B=true_B,
                               C=true_C,
                               dyn_offset=true_dyn_offset,
                               D=true_D,
                               E=true_E,
                               F=true_F,
                               lcp_offset=true_lcp_offset
                               )

# load the learned things
learned = np.load('../cartpole_debug/learned_lcs5.npy', allow_pickle=True).item()
n_state = 4
n_control = 1
n_lam = 2
F_stiffness = 1.0

# establish the VN learner (violation-based method)
gamma = learned['gamma']
epsilon = learned['epsilon']
vn_learner = lcs_learning.LCS_VN(n_state=n_state, n_control=n_control, n_lam=n_lam, F_stiffness=F_stiffness)
vn_learner.diff(gamma=gamma, epsilon=epsilon, w_C=1e-6, C_ref=0, w_F=0e-6, F_ref=0)
vn_theta = learned['vn_theta_trace'][-1]
# vn_pred_error = learned['vn_error_trace']
vn_A = vn_learner.A_fn(vn_theta)
vn_B = vn_learner.B_fn(vn_theta)
vn_C = vn_learner.C_fn(vn_theta)
vn_dyn_offset = vn_learner.dyn_offset_fn(vn_theta)
vn_D = vn_learner.D_fn(vn_theta)
vn_E = vn_learner.E_fn(vn_theta)
vn_F = vn_learner.F_fn(vn_theta)
vn_lcp_offset = vn_learner.lcp_offset_fn(vn_theta)

vn_mpc = lcs_control.LCS_MPC(A=vn_A,
                             B=vn_B,
                             C=vn_C,
                             dyn_offset=vn_dyn_offset,
                             D=vn_D,
                             E=vn_E,
                             F=vn_F,
                             lcp_offset=vn_lcp_offset
                             )

# setup the real system
true_n_state = true_sys['n_state']
true_n_control = true_sys['n_control']
true_n_lam = true_sys['n_lam']

mpc_horizon = 3
Q = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, .1, 0], [0, 0, 0, .1]])
R = 0.05
QN = scipy.linalg.solve_discrete_are(true_A, true_B, Q, R)

sim_horizon = 300
# init_state = 0.3 * np.random.randn(n_state)
init_state = np.array([0.1, 0.3, 0, 0])
true_mpc_state_traj = [init_state]
true_mpc_control_traj = []
true_mpc_lam_traj = []

st = time.time()
for t in range(sim_horizon):
    # ground truth mpc
    true_xt = true_mpc_state_traj[-1]
    # true_sol = true_mpc.mpc_penalty(true_xt, mpc_horizon, Q, R, QN, epsilon=1e-6)
    true_sol = true_mpc.mpc(true_xt, mpc_horizon, Q, R, QN)
    true_ut = true_sol['control_traj_opt'][0]
    true_mpc_control_traj += [true_ut]
    true_next_x, true_lamt = true_mpc.forward(true_xt, true_ut)
    true_mpc_state_traj += [true_next_x]
    true_mpc_lam_traj += [true_lamt]
print('time for true mpc:', time.time() - st)

vn_mpc_state_traj = [init_state]
vn_mpc_control_traj = []
vn_mpc_lam_traj = []
vn_x_pred_error = []

st = time.time()
for t in range(sim_horizon):
    # vn mpc
    vn_xt = vn_mpc_state_traj[-1]
    vn_sol = vn_mpc.mpc(vn_xt, mpc_horizon, Q, R, QN)
    vn_ut = vn_sol['control_traj_opt'][0]
    vn_mpc_control_traj += [vn_ut]
    vn_next_x, vn_lamt = true_mpc.forward(vn_xt, vn_ut)
    vn_pred_next_x, vn_pred_lamt = vn_mpc.forward(vn_xt, vn_ut)
    vn_x_pred_error += [la.norm(vn_pred_next_x - vn_next_x) / la.norm(vn_next_x)]

    vn_mpc_state_traj += [vn_next_x]
    vn_mpc_lam_traj += [vn_lamt]
print('time for vn mpc:', time.time() - st)

print(np.mean(vn_x_pred_error))



penalty_mpc_state_traj = [init_state]
penalty_mpc_control_traj = []
penalty_mpc_lam_traj = []

st = time.time()
for t in range(sim_horizon):
    # vn mpc
    penalty_xt = penalty_mpc_state_traj[-1]
    penalty_sol = vn_mpc.mpc_penalty(penalty_xt, mpc_horizon, Q, R, QN, epsilon=1e-5, )
    penalty_ut = penalty_sol['control_traj_opt'][0]
    penalty_mpc_control_traj += [penalty_ut]
    penalty_next_x, penalty_lamt = true_mpc.forward(penalty_xt, penalty_ut)
    penalty_mpc_state_traj += [penalty_next_x]
    penalty_mpc_lam_traj += [penalty_lamt]
print('time for vn penalty:', time.time() - st)

# plot the results
true_mpc_state_traj = np.array(true_mpc_state_traj)
vn_mpc_state_traj = np.array(vn_mpc_state_traj)
penalty_mpc_state_traj = np.array(penalty_mpc_state_traj)

plt.figure(figsize=(15, 3))
plt.subplot(1, 5, 1)
plt.plot(true_mpc_state_traj[:, 0], label='true_mpc')
plt.plot(vn_mpc_state_traj[:, 0], label='vn_mpc')
plt.plot(penalty_mpc_state_traj[:, 0], label='vn_mpc_penalty')
plt.legend()
plt.ylabel('x1')
plt.xlabel('sim time')

plt.subplot(1, 5, 2)
plt.plot(true_mpc_state_traj[:, 1], label='true_mpc')
plt.plot(vn_mpc_state_traj[:, 1], label='vn_mpc')
plt.plot(penalty_mpc_state_traj[:, 1], label='vn_mpc_penalty')
plt.legend()
plt.ylabel('x2')
plt.xlabel('sim time')

plt.subplot(1, 5, 3)
plt.plot(true_mpc_state_traj[:, 2], label='true_mpc')
plt.plot(vn_mpc_state_traj[:, 2], label='vn_mpc')
plt.plot(penalty_mpc_state_traj[:,2], label='vn_mpc_penalty')
plt.legend()
plt.ylabel('x3')
plt.xlabel('sim time')

plt.subplot(1, 5, 4)
plt.plot(true_mpc_state_traj[:, 3], label='true_mpc')
plt.plot(vn_mpc_state_traj[:, 3], label='vn_mpc')
plt.plot(penalty_mpc_state_traj[:, 3], label='vn_mpc_penalty')
plt.legend()
plt.ylabel('x4')
plt.xlabel('sim time')

plt.subplot(1, 5, 5)
plt.plot(true_mpc_control_traj, label='true_mpc')
plt.plot(vn_mpc_control_traj, label='vn_mpc')
plt.legend()
plt.ylabel('u')
plt.xlabel('sim time')

plt.tight_layout()

true_mpc_lam_traj = np.array(true_mpc_lam_traj)
vn_mpc_lam_traj = np.array(vn_mpc_lam_traj)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.plot(true_mpc_lam_traj[:, 0], label='true_mpc')
plt.plot(vn_mpc_lam_traj[:, 0], label='vn_mpc')
plt.legend()
plt.ylabel('lam_1')
plt.xlabel('sim time')

plt.subplot(1, 2, 2)
plt.plot(true_mpc_lam_traj[:, 1], label='true_mpc')
plt.plot(vn_mpc_lam_traj[:, 1], label='vn_mpc')
plt.legend()
plt.ylabel('lam_2')
plt.xlabel('sim time')

plt.tight_layout()
plt.show()

breakpoint()