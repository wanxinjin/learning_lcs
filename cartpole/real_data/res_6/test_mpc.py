import lcs_control
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import numpy.linalg as la

# load the system
learned_lcs = np.load('learned_lcs.npy', allow_pickle=True).item()
A = learned_lcs['A']
B = learned_lcs['B']
C = learned_lcs['C']
D = learned_lcs['D']
E = learned_lcs['E']
F = learned_lcs['F']
lcp_offset = learned_lcs['lcp_offset']

true_mpc = lcs_control.LCS_MPC(A=A,
                               B=B,
                               C=C,
                               D=D,
                               E=E,
                               F=F,
                               lcp_offset=lcp_offset
                               )

# load the learned things
n_state = 4
n_control = 1
n_lam = 2


mpc_horizon = 3
Q = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, .1, 0], [0, 0, 0, .1]])
R = 0.05
QN = scipy.linalg.solve_discrete_are(A, B, Q, R)

sim_horizon = 900
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


true_mpc_state_traj=np.array(true_mpc_state_traj)
true_mpc_lam_traj=np.array(true_mpc_lam_traj)
true_mpc_control_traj=np.array(true_mpc_control_traj)


plt.figure(figsize=(15, 3))
plt.subplot(1, 5, 1)
plt.plot(true_mpc_state_traj[:, 0], label='true_mpc')

plt.legend()
plt.ylabel('x1')
plt.xlabel('sim time')

plt.subplot(1, 5, 2)
plt.plot(true_mpc_state_traj[:, 1], label='true_mpc')

plt.legend()
plt.ylabel('x2')
plt.xlabel('sim time')

plt.subplot(1, 5, 3)
plt.plot(true_mpc_state_traj[:, 2], label='true_mpc')

plt.legend()
plt.ylabel('x3')
plt.xlabel('sim time')

plt.subplot(1, 5, 4)
plt.plot(true_mpc_state_traj[:, 3], label='true_mpc')

plt.legend()
plt.ylabel('x4')
plt.xlabel('sim time')

plt.subplot(1, 5, 5)
plt.plot(true_mpc_control_traj, label='true_mpc')
plt.legend()
plt.ylabel('u')
plt.xlabel('sim time')

plt.tight_layout()

true_mpc_lam_traj = np.array(true_mpc_lam_traj)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.plot(true_mpc_lam_traj[:, 0], label='true_mpc')
plt.legend()
plt.ylabel('lam_1')
plt.xlabel('sim time')

plt.subplot(1, 2, 2)
plt.plot(true_mpc_lam_traj[:, 1], label='true_mpc')
plt.legend()
plt.ylabel('lam_2')
plt.xlabel('sim time')

plt.tight_layout()
plt.show()

breakpoint()