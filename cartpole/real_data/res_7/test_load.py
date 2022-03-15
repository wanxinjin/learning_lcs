import numpy as np


alp_res = np.load('alp_lcs.npy', allow_pickle=True).item()
alp_A = alp_res['A']
alp_B = alp_res['B']
alp_C = alp_res['C']
alp_D = alp_res['D']
alp_E = alp_res['E']
alp_G = alp_res['G']
alp_H = alp_res['H']
alp_lcp_offset = alp_res['lcp_offset']