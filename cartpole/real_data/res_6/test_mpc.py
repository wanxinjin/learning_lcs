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

mpc = lcs_control.LCS_MPC2(A=A,
                           B=B,
                           C=C,
                           D=D,
                           E=E,
                           F=F,
                           lcp_offset=lcp_offset
                           )
mpc.oc_setup(mpc_horizon=5)


Q = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, .1, 0], [0, 0, 0, .1]])
R = 0.05
QN = scipy.linalg.solve_discrete_are(A, B, Q, R)

sim_horizon = 800
# init_state = 0.3 * np.random.randn(n_state)
init_state = np.array([0.1, 0.3, 0, 0])
true_mpc_state_traj = [init_state]
true_mpc_control_traj = []
true_mpc_lam_traj = []

previous_sol = None
time_list = []
for t in range(sim_horizon):
    # ground truth mpc
    true_xt = true_mpc_state_traj[-1]
    st = time.time()
    true_sol = mpc.mpc(true_xt, Q, R, QN, init_guess=previous_sol)
    time_list += [time.time() - st]
    true_ut = true_sol['control_traj_opt'][0]
    true_mpc_control_traj += [true_ut]
    true_next_x, true_lamt = mpc.forward(true_xt, true_ut)
    true_mpc_state_traj += [true_next_x]
    true_mpc_lam_traj += [true_lamt]
    # warm start for the next iteration
    previous_sol=true_sol
print('timing for one step:', np.array(time_list).mean())

true_mpc_state_traj = np.array(true_mpc_state_traj)
true_mpc_lam_traj = np.array(true_mpc_lam_traj)
true_mpc_control_traj = np.array(true_mpc_control_traj)

plt.figure(figsize=(15, 3))
plt.subplot(1, 5, 1)
plt.plot(true_mpc_state_traj[:, 0], label='true_mpc')

plt.legend()
plt.ylabel('x1')
plt.xlabel('sim time')

plt.subplot(1, 5, 2)
plt.plot(true_mpc_state_traj[:, 1])

plt.legend()
plt.ylabel('x2')
plt.xlabel('sim time')

plt.subplot(1, 5, 3)
plt.plot(true_mpc_state_traj[:, 2])

plt.legend()
plt.ylabel('x3')
plt.xlabel('sim time')

plt.subplot(1, 5, 4)
plt.plot(true_mpc_state_traj[:, 3])

plt.legend()
plt.ylabel('x4')
plt.xlabel('sim time')

plt.subplot(1, 5, 5)
plt.plot(true_mpc_control_traj)
plt.legend()
plt.ylabel('u')
plt.xlabel('sim time')

plt.tight_layout()

true_mpc_lam_traj = np.array(true_mpc_lam_traj)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.plot(true_mpc_lam_traj[:, 0])
plt.legend()
plt.ylabel('lam_1')
plt.xlabel('sim time')

plt.subplot(1, 2, 2)
plt.plot(true_mpc_lam_traj[:, 1])
plt.legend()
plt.ylabel('lam_2')
plt.xlabel('sim time')

plt.tight_layout()
plt.show()

breakpoint()
