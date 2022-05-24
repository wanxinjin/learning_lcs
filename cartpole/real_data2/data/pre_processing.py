import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter

win_len = 11
order = 3

# ================================= left impact
train_x_batch = []
train_u_batch = []
train_x_next_batch = []
# left trial 1
load = sio.loadmat('trial1.mat')
input = load['input'].flatten()
friction_force = load['Friction_force'].flatten()
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()

# position_cart = savgol_filter(position_cart, win_len, order)
# position_pole = savgol_filter(position_pole, win_len, order)
velocity_cart = savgol_filter(velocity_cart, win_len, order)
velocity_pole = savgol_filter(velocity_pole, win_len, order)
control = savgol_filter(control, win_len, order)

state = np.vstack((position_cart, position_pole, velocity_cart, velocity_pole)).T
# do the cut
start = 400
end = 1200
state = state[start:end + 1]
control = control[start: end]

# store
train_x_batch += [state[0:-1]]
train_u_batch += [control]
train_x_next_batch += [state[1:]]

# left trial 2
load = sio.loadmat('trial2.mat')
input = load['input'].flatten()
friction_force = load['Friction_force'].flatten()
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()

velocity_cart = savgol_filter(velocity_cart, win_len, order)
velocity_pole = savgol_filter(velocity_pole, win_len, order)
control = savgol_filter(control, win_len, order)

state = np.vstack((position_cart, position_pole, velocity_cart, velocity_pole)).T
# do the cut
start = 400
end = 1200
state = state[start:end + 1]
control = control[start: end]

# store
train_x_batch += [state[0:-1]]
train_u_batch += [control]
train_x_next_batch += [state[1:]]

# left trial 3
load = sio.loadmat('trial3.mat')
input = load['input'].flatten()
friction_force = load['Friction_force'].flatten()
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()

velocity_cart = savgol_filter(velocity_cart, win_len, order)
velocity_pole = savgol_filter(velocity_pole, win_len, order)
control = savgol_filter(control, win_len, order)

state = np.vstack((position_cart, position_pole, velocity_cart, velocity_pole)).T
# do the cut
start = 400
end = 1200
state = state[start:end + 1]
control = control[start: end]

# store
train_x_batch += [state[0:-1]]
train_u_batch += [control]
train_x_next_batch += [state[1:]]

# left trial 4
load = sio.loadmat('trial4.mat')
input = load['input'].flatten()
friction_force = load['Friction_force'].flatten()
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()

velocity_cart = savgol_filter(velocity_cart, win_len, order)
velocity_pole = savgol_filter(velocity_pole, win_len, order)
control = savgol_filter(control, win_len, order)

state = np.vstack((position_cart, position_pole, velocity_cart, velocity_pole)).T
# do the cut
start = 400
end = 1200
state = state[start:end + 1]
control = control[start: end]

# store
train_x_batch += [state[0:-1]]
train_u_batch += [control]
train_x_next_batch += [state[1:]]

# ================================= right impact


# right trial 1
load = sio.loadmat('trial6.mat')
input = load['input'].flatten()
friction_force = load['Friction_force'].flatten()
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()

velocity_cart = savgol_filter(velocity_cart, win_len, order)
velocity_pole = savgol_filter(velocity_pole, win_len, order)
control = savgol_filter(control, win_len, order)

state = np.vstack((position_cart, position_pole, velocity_cart, velocity_pole)).T
# do the cut
start = 400
end = 1200
state = state[start:end + 1]
control = control[start: end]

# store
train_x_batch += [state[0:-1]]
train_u_batch += [control]
train_x_next_batch += [state[1:]]

# right trial 2
load = sio.loadmat('trial7.mat')
input = load['input'].flatten()
friction_force = load['Friction_force'].flatten()
# friction_force=0
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()

velocity_cart = savgol_filter(velocity_cart, win_len, order)
velocity_pole = savgol_filter(velocity_pole, win_len, order)
control = savgol_filter(control, win_len, order)

state = np.vstack((position_cart, position_pole, velocity_cart, velocity_pole)).T
# do the cut
start = 400
end = 1200
state = state[start:end + 1]
control = control[start: end]

# store
train_x_batch += [state[0:-1]]
train_u_batch += [control]
train_x_next_batch += [state[1:]]

# right trial 3
load = sio.loadmat('trial8.mat')
input = load['input'].flatten()
friction_force = load['Friction_force'].flatten()
# friction_force=0
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()

velocity_cart = savgol_filter(velocity_cart, win_len, order)
velocity_pole = savgol_filter(velocity_pole, win_len, order)
control = savgol_filter(control, win_len, order)

state = np.vstack((position_cart, position_pole, velocity_cart, velocity_pole)).T
# do the cut
start = 400
end = 1200
state = state[start:end + 1]
control = control[start: end]

# store
train_x_batch += [state[0:-1]]
train_u_batch += [control]
train_x_next_batch += [state[1:]]

# right trial 4
load = sio.loadmat('trial9.mat')
input = load['input'].flatten()
friction_force = load['Friction_force'].flatten()
# friction_force=0
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()

velocity_cart = savgol_filter(velocity_cart, win_len, order)
velocity_pole = savgol_filter(velocity_pole, win_len, order)
control = savgol_filter(control, win_len, order)

state = np.vstack((position_cart, position_pole, velocity_cart, velocity_pole)).T
# do the cut
start = 400
end = 1200
state = state[start:end + 1]
control = control[start: end]

# store
train_x_batch += [state[0:-1]]
train_u_batch += [control]
train_x_next_batch += [state[1:]]

# reorgnize
train_x_batch = np.vstack(train_x_batch)
train_x_next_batch = np.vstack(train_x_next_batch)
train_u_batch = np.hstack(train_u_batch)

print(train_u_batch.shape)
print(train_x_batch.shape)
print(train_x_next_batch.shape)


params = {'axes.labelsize': 25,
          'axes.titlesize': 25,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'legend.fontsize': 20}
plt.rcParams.update(params)

# plt.plot(train_x_batch[:,0], label='cart pos')
# plt.plot(train_x_batch[:,1], label='pole pos')
xvars=np.arange(len(train_x_batch))*0.01
plt.plot(xvars, train_x_batch[:, 2], label='cart vel', lw=3)
plt.plot(xvars, train_x_batch[:, 3], label='pole vel', lw=3)
plt.xlabel('time [s]')
plt.ylabel('velocity [m/s], [rad/s]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



# save
n_state = 4
n_control = 1
n_lam = 2
np.save('train_data_2.npy', {'train_x': train_x_batch,
                             'train_u': train_u_batch,
                             'train_x_next': train_x_next_batch,
                             'n_state': n_state,
                             'n_control': n_control,
                             'n_lam': n_lam
                             })
