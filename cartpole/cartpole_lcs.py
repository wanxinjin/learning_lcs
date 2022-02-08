import time

from lcs import lcs_learning
import numpy as np
from casadi import *
import lcs.optim as opt
from lcs import lcs_control
import matplotlib.pyplot as plt

# --------
g = 9.81
mp = 0.411
mc = 0.978
len_p = 0.6
len_com = 0.4267
d1 = 0.35
d2 = -0.35
ks = 100
Ts = 0.01

n_state = 4
n_control = 1
n_lam = 2

A = [[0, 0, 1, 0], [0, 0, 0, 1], [0, g * mp / mc, 0, 0], [0, g * (mc + mp) / (len_com * mc), 0, 0]]
A = np.asarray(A)
B = [[0], [0], [1 / mc], [1 / (len_com * mc)]]
B = np.asarray(B)
C = [[0, 0], [0, 0], [(-1 / mc) + (len_p / (mc * len_com)), (1 / mc) - (len_p / (mc * len_com))],
     [(-1 / (mc * len_com)) + (len_p * (mc + mp)) / (mc * mp * len_com * len_com),
      -((-1 / (mc * len_com)) + (len_p * (mc + mp)) / (mc * mp * len_com * len_com))]]
C = np.asarray(C)
D = [[-1, len_p, 0, 0], [1, -len_p, 0, 0]]
D = np.asarray(D)
F = 1 / ks * np.eye(2)
F = np.asarray(F)
lcp_offset = [[d1], [-d2]]
lcp_offset = np.asarray(lcp_offset)
E = np.zeros((2, 1))

A = np.eye(n_state) + Ts * A
B = Ts * B
C = Ts * C

D = D
E = E
F = F
H = np.zeros((n_lam, n_lam))
G=np.sqrt(F)

# form the lcs system
min_sig = min(np.linalg.eigvals(F + F.T))


lcs_mats = {
    'n_state': n_state,
    'n_control': n_control,
    'n_lam': n_lam,
    'A': A,
    'B': B,
    'C': C,
    'D': D,
    'E': E,
    'G': G,
    'H': H,
    'F': F,
    'lcp_offset': lcp_offset,
    'min_sig': min_sig}

np.save('cartpole_lcs', lcs_mats)
