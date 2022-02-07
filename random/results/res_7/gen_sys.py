import test_class
import numpy as np

# generate the data
n_state = 3
n_lam = 3
# np.random.seed(1)
A = np.random.randn(n_state, n_state)
C = np.random.randn(n_state, n_lam)

lcp_offset = np.random.randn(n_lam)

D = np.random.randn(n_lam, n_state)
G = 1 * np.random.randn(n_lam, n_lam)
F = G @ G.T
# form the lcs system
min_sig = min(np.linalg.eigvals(F))
print(min_sig)

lcs_mats = {
    'n_state': n_state,
    'n_lam': n_lam,
    'A': A,
    'C': C,
    'D': D,
    'G': G,
    'lcp_offset': lcp_offset,
    'F': F,
    'min_sig': min_sig}

np.save('random_lcs', lcs_mats)
