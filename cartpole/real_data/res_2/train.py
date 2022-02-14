import numpy as np

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

# ==============================   load the real training data   ==================================
train_data = np.load('../data/train_data.npy', allow_pickle=True).item()
n_state = train_data['n_state']
n_control = train_data['n_control']
n_lam = train_data['n_lam']

train_x_batch = train_data['train_x']
train_u_batch = train_data['train_u']
train_x_next_batch = train_data['train_x_next']
train_lam_batch = train_data['train_lam']
train_data_size = train_x_batch.shape[0]

train_mode_list, train_mode_frequency_list = cartpole_class.statiModes(train_lam_batch)
print('train_data size:', train_data_size)
print('number of modes in the training data:', train_mode_frequency_list.size)
print('mode frequency in the training data: ', train_mode_frequency_list)
# check the mode index
train_mode_list, train_mode_indices = cartpole_class.plotModes(train_lam_batch)

# =============== plot the training data, each color for each mode  ======================================

plt.ion()

# here we are creating sub plots
figure, ax = plt.subplots(figsize=(10, 8))
ax.plot(train_lam_batch[:,0], label='true lam 1', linestyle='--')
ax.plot(train_lam_batch[:,1], label='true lam 2', linestyle='--')
line1, = ax.plot(np.zeros_like(train_lam_batch[:,0]), label='learned lam 1',)
line2, = ax.plot(np.zeros_like(train_lam_batch[:,1]), label='learned lam 2')
ax.legend()

# ==============================   create the learner object    ========================================
learner = cartpole_class.cartpole_learner2(n_state, n_control, n_lam=n_lam,
                                           stiffness=1.)
# ================================   beginning the training process    ======================================
# doing learning process
curr_theta = 0.5 * np.random.rand(learner.n_theta)
# curr_theta = true_theta + 0.5 * np.random.randn(learner.n_theta)
mini_batch_size = train_data_size
loss_trace = []
theta_trace = []
optimizier = opt.Adam()
optimizier.learning_rate = 1e-1
epsilon = np.logspace(2, -3, 10000)
for k in range(10000):
    # mini batch dataset
    shuffle_index = np.random.permutation(train_data_size)[0:mini_batch_size]
    x_mini_batch = train_x_batch[shuffle_index]
    u_mini_batch = train_u_batch[shuffle_index]
    x_next_mini_batch = train_x_next_batch[shuffle_index]
    lam_mini_batch = train_lam_batch[shuffle_index]

    # compute the lambda batch
    learner.differetiable(epsilon=epsilon[k])
    lam_phi_opt_mini_batch, loss_opt_batch = learner.compute_lambda(x_mini_batch, u_mini_batch, lam_mini_batch,
                                                                    curr_theta)

    # compute the gradient
    dtheta, loss, dyn_loss, lcp_loss = \
        learner.gradient_step(x_mini_batch, u_mini_batch, lam_mini_batch, curr_theta, lam_phi_opt_mini_batch)

    # store and update
    loss_trace += [loss]
    theta_trace += [curr_theta]
    curr_theta = optimizier.step(curr_theta, dtheta)

    if k % 100 == 0:
        # on the prediction using the current learned lcs
        pred_lam_batch = learner.dyn_prediction(train_x_batch, train_u_batch, curr_theta)

        # compute the prediction error
        error_lam_batch = pred_lam_batch - train_lam_batch
        error = la.norm(error_lam_batch, axis=1)
        max_error=np.amax(error)

        # compute the predicted mode statistics
        pred_mode_list, pred_mode_indices = cartpole_class.plotModes(pred_lam_batch)

        # updating data values
        line1.set_ydata(pred_lam_batch[:,0])
        line2.set_ydata(pred_lam_batch[:,1])

        # drawing updated values
        figure.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        figure.canvas.flush_events()

        plt.pause(1)



        print(
            k,
            '| loss:', loss,
            '| grad:', norm_2(dtheta),
            '| dyn:', dyn_loss,
            '| lcp:', lcp_loss,
            '| max_error:', max_error,
            '| PMC:', len(pred_mode_list),
            '| Epsilon:', epsilon[k],
        )


# on the prediction using the current learned lcs
pred_lam_batch = learner.dyn_prediction(train_x_batch, train_u_batch, curr_theta)

# compute the prediction error
error_lam_batch = pred_lam_batch - train_lam_batch
relative_error = (la.norm(error_lam_batch, axis=1) / (la.norm(train_lam_batch, axis=1) + 0.0001)).mean()
error = la.norm(error_lam_batch, axis=1).mean()



# save
np.save('learned', {
    'theta_trace': theta_trace,
    'loss_trace': loss_trace,
    'color_list': color_list,
    'train_lam_batch': train_lam_batch,
    'pred_lam_batch': pred_lam_batch,
})


