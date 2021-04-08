import numpy as np
from matplotlib import pyplot as plt

ROLL = 20171158
Arr = [8.8, 9, 10, 11, 12, 13, 13.9, 15, 16, 16.5, 16.6, 17, 18, 19, 20]

"""
C(0)   R(1)
A(2)   B(3)
"""

A, B, C, R = 0, 1, 2, 3
Gamma = 0.2
Delta = 0.01

V = np.zeros((4,), dtype=np.float)

p = np.array([
    # A     B      C     R
    [[0.2, 0.8, 0.00, 0.00],  # A, r
     [0.2, 0.0, 0.80, 0.00],  # A, u
     ],
    [[0.8, 0.2, 0.00, 0.00],  # B, l
     [0.0, 0.2, 0.00, 0.80],  # B, u
     ],
    [[0.0, 0.0, 0.75, 0.25],  # C, r
     [0.8, 0.0, 0.20, 0.00],  # C, d
     ],
])

# step costs
r0 = np.array([
    # A   B   C   R
    [-1, -1, -1,  0],  # A (a->r is an invalid move)
    [-1, -1,  0, -4],  # B (b->c is an invalid move)
    [-1,  0, -1, -3],  # C (c->b is an invalid move)
    [0.,  0., 0,  0],  # R (r->* is an invalid move)
])

# dirs
Up, Le = 1, 0
Ri, Do = 0, 1

# move_links = np.array([
#     # 0 1
#     [B, C],  # A
#     [A, R],  # B
#     [R, A],  # C
#     [R, R],  # R
# ])

directions = np.array([
    ["Right", "Up"],  # A
    ["Left", "Up"],  # B
    ["Right", "Down"],  # C
])


def update_v(state: int, name, oldv):
    # p(R|C, r) is Pt[C][Ri][R]
    # r0(C, r, R) is r0[C][R] action can be ignored for now
    rs = np.zeros((2,), dtype=np.float)
    fxs = np.zeros((2,), dtype=np.float)
    print("For ", name, ":", sep="")
    for i in range(2):
        rs[i] = np.sum(p[state][i] * r0[state])
        fxs[i] = Gamma * np.sum(oldv * p[state][i])
        insum = [f"({x})({y})" for x, y in zip(oldv, p[state][i]) if y != 0]
        gammaterm = f"+ r ({' + '.join(insum)})"
        print(f"""
\tR({name},{directions[state][i]}) {gammaterm}
\t\t= {np.round(rs[i], 5)} + {Gamma} ({np.sum(oldv * p[state][i])})
\t\t= {np.round(rs[i] + fxs[i], 9)}""")
    lst = rs + fxs
    maxcv = np.max(lst)
    print(f"\t\t=> max({lst[0]:.8f}, {lst[1]:.8f})")
    V[state] = maxcv
    V[:] = np.round(V, 9)


def print_v(v, i, name, idx):
    print(f"\t\t=> V{i}({name}) = {v[idx]}")


def print_v_diff(v, oldv, i):
    ad = abs(v[A]-oldv[A])
    bd = abs(v[B]-oldv[B])
    cd = abs(v[C]-oldv[C])
    print(f"\n\n\t|V{i}(A) - V{i-1}(A)| = {ad:f}")
    print(f"\t|V{i}(B) - V{i-1}(B)| = {bd:f}")
    print(f"\t|V{i}(C) - V{i-1}(C)| = {cd:f}")
    strs = []
    opstrs = []
    if ad > Delta:
        strs.append(f"|V{i}(A) - V{i-1}(A)|")
    else:
        opstrs.append(f"|V{i}(A) - V{i-1}(A)| < {Delta}")
    if bd > Delta:
        strs.append(f"|V{i}(B) - V{i-1}(B)|")
    else:
        opstrs.append(f"|V{i}(B) - V{i-1}(B)| < {Delta}")
    if cd > Delta:
        strs.append(f"|V{i}(C) - V{i-1}(C)|")
    else:
        opstrs.append(f"|V{i}(C) - V{i-1}(C)| < {Delta}")

    print()
    if len(strs) > 0:
        print("\tBut here,\n\t\t"+" and ".join(strs), f"> {Delta} => didn't converge, so next iteration")
    else:
        print("\tHere,\n\t\t"+" and ".join(opstrs), "=> converged!")


def reward_based(reward = Arr[ROLL % 15]):
    global V
    V = np.zeros((4,), dtype=np.float)
    V[R] = reward

    i = 0
    while True:
        i += 1
        print("\niteration=", i, sep='')
        print("_"*10+"\n")
        V_bef = V.copy()
        update_v(A, "A", V_bef)
        print_v(V, i, "A", A)
        update_v(B, "B", V_bef)
        print_v(V, i, "B", B)
        update_v(C, "C", V_bef)
        print_v(V, i, "C", C)
        print_v_diff(V, V_bef, i)
        if np.all(np.abs(V-V_bef) < Delta) or i > 10:
            # print("\niteration=", i, sep='')
            print_v(V, i, "A", A)
            print_v(V, i, "B", B)
            print_v(V, i, "C", C)
            if V[B] > V[C]:
                print("B > C", reward)
            else:
                print("B < C", reward)
            break

if __name__ == "__main__":
    reward_based()
    # plot values with increasing reward
    # avals = []
    # bvals = []
    # cvals = []
    # for reward in Arr:
        # reward_based(reward)
        # avals.append(V[A])
        # bvals.append(V[B])
        # cvals.append(V[C])
    # plt.plot(Arr, avals, label="A")
    # plt.plot(Arr, bvals, label="B")
    # plt.plot(Arr, cvals, label="C")
    # plt.legend()
    # plt.show()
