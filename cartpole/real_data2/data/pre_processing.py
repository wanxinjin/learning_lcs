import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

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
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()
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
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()
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
control = input + friction_force
position_cart = load['position_cart'].flatten()
position_pole = load['position_pole'].flatten()
velocity_cart = load['CartVel'].flatten()
velocity_pole = load['PoleVel'].flatten()
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

#
# plt.plot(train_x_batch[:,0])
# plt.plot(train_x_batch[:,1])
# plt.plot(train_x_batch[:,2])
# plt.plot(train_x_batch[:,3])
# plt.show()
#
# plt.figure()
# plt.plot(train_u_batch)
# plt.show()
