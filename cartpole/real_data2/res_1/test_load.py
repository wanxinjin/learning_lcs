import numpy as np

learned_lcs = np.load('learned_lcs.npy', allow_pickle=True).item()
A = learned_lcs['A']
B = learned_lcs['B']
C = learned_lcs['C']
D = learned_lcs['D']
E = learned_lcs['E']
F = learned_lcs['F']
lcp_offset = learned_lcs['lcp_offset']

print('------------------------------------------------')
print('A')
print(A)
print('------------------------------------------------')
print('B')
print(B)
print('------------------------------------------------')
print('C')
print(C)
print('------------------------------------------------')
print('D')
print(D)
print('------------------------------------------------')
print('E')
print(E)
print('------------------------------------------------')
print('F')
print(F)
print('------------------------------------------------')
print('lcp_offset')
print(lcp_offset)

np.save('wanxin_mats.npy', {'A': A,
                            'B': B,
                            'C': C,
                            'D': D,
                            'E': E,
                            'F': F,
                            'lcp_offset': lcp_offset
                            })
