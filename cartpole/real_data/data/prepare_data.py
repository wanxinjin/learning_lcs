import numpy as np
import matplotlib.pyplot as plt

# ================================= left impact
# left impact 1
impact_left1 = np.load('data/impact_left1.npy', allow_pickle=True).item()
traj_x_l1 = impact_left1['state_traj'][20:81]
traj_u_l1 = impact_left1['input_traj'][20:80]
traj_lam_l1 = impact_left1['lam_traj'][20:81]

train_x_l1 = traj_x_l1[0:-1]
train_u_l1 = traj_u_l1
train_x_next_l1 = traj_x_l1[1:]
train_lam_l1 = traj_lam_l1[0:-1]


# left impact 2
impact_left2 = np.load('data/impact_left2.npy', allow_pickle=True).item()
traj_x_l2 = impact_left2['state_traj'][40:91]
traj_u_l2 = impact_left2['input_traj'][40:90]
traj_lam_l2 = impact_left2['lam_traj'][40:91]

train_x_l2 = traj_x_l2[0:-1]
train_u_l2 = traj_u_l2
train_x_next_l2 = traj_x_l2[1:]
train_lam_l2 = traj_lam_l2[0:-1]



# left impact 3
impact_left3 = np.load('data/impact_left3.npy', allow_pickle=True).item()
traj_x_l3 = impact_left3['state_traj'][1:50]
traj_u_l3 = impact_left3['input_traj'][1:50]
traj_lam_l3 = impact_left3['lam_traj'][1:50]

train_x_l3 = traj_x_l3[0:-1]
train_u_l3 = traj_u_l3[0:-1]
train_x_next_l3 = traj_x_l3[1:]
train_lam_l3 = traj_lam_l3[0:-1]


# training data on the left side
train_x_l = np.vstack((train_x_l1, train_x_l2, train_x_l3))
train_u_l = np.vstack((train_u_l1, train_u_l2, train_u_l3))
train_x_next_l = np.vstack((train_x_next_l1, train_x_next_l2, train_x_next_l3))
train_lam_l = np.vstack((train_lam_l1, train_lam_l2, train_lam_l3))

# left impact 4
impact_left4 = np.load('data/impact_left4.npy', allow_pickle=True).item()
traj_x_l4 = impact_left4['state_traj']
traj_u_l4 = impact_left4['input_traj']
traj_lam_l4 = impact_left4['lam_traj']

# testing data on the left
test_x_l = traj_x_l4[0:-1]
test_u_l = traj_u_l4
test_x_next_l = traj_x_l4[1:]
test_lam_l = traj_lam_l4[0:-1]

# ============================= right impact
# right impact 1
impact_right1 = np.load('data/impact_right1.npy', allow_pickle=True).item()
traj_x_r1 = impact_right1['state_traj'][30:80]
traj_u_r1 = impact_right1['input_traj'][30:79]
traj_lam_r1 = impact_right1['lam_traj'][30:80]

train_x_r1 = traj_x_r1[0:-1]
train_u_r1 = traj_u_r1
train_x_next_r1 = traj_x_r1[1:]
train_lam_r1 = traj_lam_r1[0:-1]




# left impact 2
impact_right2 = np.load('data/impact_right2.npy', allow_pickle=True).item()
traj_x_r2 = impact_right2['state_traj'][20:61]
traj_u_r2 = impact_right2['input_traj'][20:60]
traj_lam_r2 = impact_right2['lam_traj'][20:60]

train_x_r2 = traj_x_r2[0:-1]
train_u_r2 = traj_u_r2
train_x_next_r2 = traj_x_r2[1:]
train_lam_r2 = traj_lam_r2[0:]





# left impact 3
impact_right3 = np.load('data/impact_right3.npy', allow_pickle=True).item()
traj_x_r3 = impact_right3['state_traj'][20:60]
traj_u_r3 = impact_right3['input_traj'][20:59]
traj_lam_r3 = impact_right3['lam_traj'][20:60]

train_x_r3 = traj_x_r3[0:-1]
train_u_r3 = traj_u_r3
train_x_next_r3 = traj_x_r3[1:]
train_lam_r3 = traj_lam_r3[0:-1]






# training data on the left side
train_x_r = np.vstack((train_x_r1, train_x_r2, train_x_r3))
train_u_r = np.vstack((train_u_r1, train_u_r2, train_u_r3))
train_x_next_r = np.vstack((train_x_next_r1, train_x_next_r2, train_x_next_r3))
train_lam_r = np.vstack((train_lam_r1, train_lam_r2, train_lam_r3))

# left impact 4
impact_right4 = np.load('data/impact_right4.npy', allow_pickle=True).item()
traj_x_r4 = impact_right4['state_traj']
traj_u_r4 = impact_right4['input_traj']
traj_lam_r4 = impact_right4['lam_traj']

# testing data on the left
test_x_r = traj_x_r4[0:-1]
test_u_r = traj_u_r4
test_x_next_r = traj_x_r4[1:]
test_lam_r = traj_lam_r4[0:-1]

# save
train_x = np.vstack((train_x_l, train_x_r))
train_u = np.vstack((train_u_l, train_u_r))
train_x_next = np.vstack((train_x_next_l, train_x_next_r))
train_lam = np.vstack((train_lam_l, train_lam_r))

test_x = np.vstack((test_x_l, test_x_r))
test_u = np.vstack((test_u_l, test_u_r))
test_x_next = np.vstack((test_x_next_l, test_x_next_r))
test_lam = np.vstack((test_lam_l, test_lam_r))



# plt.plot(train_lam[:,0])
# plt.plot(train_lam[:,1])
# plt.show()

#
# print(train_x_r3.shape)
# print(train_u_r3.shape)
# print(train_x_next_r3.shape)
# print(train_lam_r3.shape)
#
# plt.plot(train_lam_r3)
# plt.show()
# breakpoint()


np.save('data/train_data',
        {
            'n_state': 4,
            'n_control': 1,
            'n_lam': 2,
            'train_x': train_x,
            'train_u': train_u,
            'train_x_next': train_x_next,
            'train_lam': train_lam,
        }
        )

np.save('data/test_data',
        {
            'n_state': 4,
            'n_control': 1,
            'n_lam': 2,
            'test_x': test_x,
            'test_u': test_u,
            'test_x_next': test_x_next,
            'test_lam': test_lam,
        })
