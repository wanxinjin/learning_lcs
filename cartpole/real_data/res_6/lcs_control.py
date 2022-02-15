import time
import casadi
import numpy as np
from casadi import *
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

stiffness = 1


class LCS_OC:
    def __init__(self, A, B, C, dyn_offset, D, E, F, lcp_offset):
        self.A = DM(A)
        self.B = DM(B)
        self.C = DM(C)
        self.dyn_offset = DM(dyn_offset)
        self.D = DM(D)
        self.E = DM(E)
        self.F = DM(F)
        self.lcp_offset = DM(lcp_offset)

        self.n_state = self.A.shape[0]
        self.n_control = self.B.shape[1]
        self.n_lam = self.C.shape[1]

        # define the system variable
        x = casadi.MX.sym('x', self.n_state)
        u = casadi.MX.sym('u', self.n_control)
        xu_pair = vertcat(x, u)
        lam = casadi.MX.sym('lam', self.n_lam)

        # dynamics
        dyn = self.A @ x + self.B @ u + self.C @ lam + self.dyn_offset
        self.dyn_fn = Function('dyn_fn', [xu_pair, lam], [dyn])

        # loss function
        lcp_loss = dot(self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset, lam)

        # constraints
        dis_cstr = self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset
        lam_cstr = lam
        total_cstr = vertcat(dis_cstr, lam_cstr)
        self.dis_cstr_fn = Function('dis_cstr_fn', [lam, xu_pair], [dis_cstr])

        # establish the qp solver to solve for LCP
        quadprog = {'x': lam, 'f': lcp_loss, 'g': total_cstr, 'p': xu_pair}
        opts = {'printLevel': 'none', }
        self.qpSolver = qpsol('S', 'qpoases', quadprog, opts)

    def solve(self, state_int, horizon, mat_Q, mat_R):
        # set initial condition
        init_state = casadi.DM(state_int).full().flatten().tolist()
        Q = DM(mat_Q)
        R = DM(mat_R)

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = casadi.MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += init_state
        ubw += init_state
        w0 += init_state

        # formulate the NLP
        for k in range(horizon):
            # New NLP variable for the control
            Uk = casadi.MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0]

            # new NLP variable for the complementarity variable
            Lamk = casadi.MX.sym('lam' + str(k), self.n_lam)
            w += [Lamk]
            lbw += self.n_lam * [0]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0]

            # Add complementarity equation
            g += [self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset]
            lbg += self.n_lam * [0]
            ubg += self.n_lam * [inf]

            g += [casadi.dot(self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset, Lamk)]
            lbg += [0]
            ubg += [0]

            # Integrate till the end of the interval
            Xnext = self.A @ Xk + self.B @ Uk + self.C @ Lamk + self.dyn_offset
            Ck = dot(Xk, Q @ Xk) + dot(Uk, R @ Uk)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Add the final cost
        J = J + dot(Xk, Q @ Xk)

        # Create an NLP solver and solve
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g)}
        solver = casadi.nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x']
        g = sol['g']

        # extract the optimal control and state
        sol_traj = w_opt[0:horizon * (self.n_state + self.n_control + self.n_lam)].reshape(
            (self.n_state + self.n_control + self.n_lam, -1))
        x_traj = casadi.horzcat(sol_traj[0:self.n_state, :],
                                w_opt[horizon * (self.n_state + self.n_control + self.n_lam):]).T
        u_traj = sol_traj[self.n_state:self.n_state + self.n_control, :].T
        lam_traj = sol_traj[self.n_state + self.n_control:, :].T

        opt_sol = {'state_traj_opt': x_traj.full(),
                   'control_traj_opt': u_traj.full(),
                   'lam_traj_opt': lam_traj.full()}

        return opt_sol

    def forward(self, xt, ut):
        # compute the current lam
        xu_t = vertcat(xt, ut)
        sol = self.qpSolver(lbg=0., p=xu_t)
        lamt_t = sol['x'].full().flatten()
        x_next = self.dyn_fn(xu_t, lamt_t).full().flatten()

        return x_next, lamt_t

        # end

    def randSimulation(self, state_init, horizon):

        state_traj = [state_init]
        lam_traj = []
        control_traj = np.random.randn(horizon, self.n_control)
        for t in range(horizon):
            u_t = control_traj[t]
            x_t = state_traj[-1]
            x_next, lam_t = self.forward(u_t, x_t)
            state_traj += [x_next]
            lam_traj += [lam_t]

        sol = {'state_traj_opt': np.array(state_traj),
               'lam_traj_opt': np.array(lam_traj),
               'control_traj_opt': control_traj}

        return sol


class LCS_MPC:
    def __init__(self, A, B, C, D, E, F, lcp_offset):
        self.A = DM(A)
        self.B = DM(B)
        self.C = DM(C)
        self.D = DM(D)
        self.E = DM(E)
        self.F = DM(F)
        self.lcp_offset = DM(lcp_offset)

        self.n_state = self.A.shape[0]
        self.n_control = self.B.shape[1]
        self.n_lam = self.C.shape[1]

        # define the system variable
        x = casadi.MX.sym('x', self.n_state)
        u = casadi.MX.sym('u', self.n_control)
        xu_pair = vertcat(x, u)
        lam = casadi.MX.sym('lam', self.n_lam)

        # dynamics
        dyn = self.A @ x + self.B @ u + self.C @ lam
        self.dyn_fn = Function('dyn_fn', [xu_pair, lam], [dyn])

        # loss function
        lcp_loss = dot(self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset, lam)

        # constraints
        dis_cstr = self.D @ x + self.E @ u + self.F @ lam + self.lcp_offset
        lam_cstr = lam
        total_cstr = vertcat(dis_cstr, lam_cstr)
        self.dis_cstr_fn = Function('dis_cstr_fn', [lam, xu_pair], [dis_cstr])

        # establish the qp solver to solve for LCP
        quadprog = {'x': lam, 'f': lcp_loss, 'g': total_cstr, 'p': xu_pair}
        opts = {'printLevel': 'none', }
        self.lcpSolver = qpsol('S', 'qpoases', quadprog, opts)

    def forward(self, x_t, u_t):
        xu_pair = vertcat(DM(x_t), DM(u_t))
        sol = self.lcpSolver(p=xu_pair, lbg=0.)
        lam_t = sol['x'].full().flatten()
        x_next = self.dyn_fn(xu_pair, lam_t).full().flatten()
        return x_next, lam_t

    def mpc(self, init_state, horizon, mat_Q, mat_R, mat_QN):
        # set initial condition
        init_state = casadi.DM(init_state).full().flatten().tolist()
        Q = DM(mat_Q)
        R = DM(mat_R)
        QN = DM(mat_QN)

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = casadi.MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += init_state
        ubw += init_state
        w0 += init_state

        # formulate the NLP
        for k in range(horizon):
            # New NLP variable for the control
            Uk = casadi.MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0]

            # new NLP variable for the complementarity variable
            Lamk = casadi.MX.sym('lam' + str(k), self.n_lam)
            w += [Lamk]
            lbw += self.n_lam * [0]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0]

            # Add complementarity equation
            g += [self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset]
            lbg += self.n_lam * [0]
            ubg += self.n_lam * [inf]

            g += [casadi.dot(self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset, Lamk)]
            lbg += [0]
            ubg += [0]

            # Integrate till the end of the interval
            Xnext = self.A @ Xk + self.B @ Uk + self.C @ Lamk
            Ck = dot(Xk, Q @ Xk) + dot(Uk, R @ Uk)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Add the final cost
        J = J + dot(Xk, QN @ Xk)

        # Create an NLP solver and solve
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g)}
        solver = casadi.nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x']
        g = sol['g']

        # extract the optimal control and state
        sol_traj = w_opt[0:horizon * (self.n_state + self.n_control + self.n_lam)].reshape(
            (self.n_state + self.n_control + self.n_lam, -1))
        x_traj = casadi.horzcat(sol_traj[0:self.n_state, :],
                                w_opt[horizon * (self.n_state + self.n_control + self.n_lam):]).T
        u_traj = sol_traj[self.n_state:self.n_state + self.n_control, :].T
        lam_traj = sol_traj[self.n_state + self.n_control:, :].T

        opt_sol = {'state_traj_opt': x_traj.full(),
                   'control_traj_opt': u_traj.full(),
                   'lam_traj_opt': lam_traj.full()}

        return opt_sol

    def mpc_penalty(self, init_state, horizon, mat_Q, mat_R, mat_QN, epsilon=0.001, gamma=1e-2):
        # set initial condition
        init_state = casadi.DM(init_state).full().flatten().tolist()
        Q = DM(mat_Q)
        R = DM(mat_R)
        QN = DM(mat_QN)

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = casadi.MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += init_state
        ubw += init_state
        w0 += init_state

        # formulate the NLP
        for k in range(horizon):
            # New NLP variable for the control
            Uk = casadi.MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0]
            # new NLP variable for the complementarity variable
            Lamk = casadi.MX.sym('Lam_' + str(k), self.n_lam)
            w += [Lamk]
            lbw += self.n_lam * [0]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0]
            Phik = casadi.MX.sym('Phi_' + str(k), self.n_lam)
            w += [Phik]
            lbw += self.n_lam * [0]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0]

            # Add complementarity equation
            lcp_penalty = dot(Lamk * Lamk, Phik * Phik) + \
                          dot(self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset - Phik,
                              self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset - Phik)

            # Integrate till the end of the interval
            Xnext = self.A @ Xk + self.B @ Uk + self.C @ Lamk
            Ck = dot(Xk, Q @ Xk) + dot(Uk, R @ Uk) + 1 / epsilon * lcp_penalty
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Add the final cost
        J = J + dot(Xk, QN @ Xk)

        # Create an NLP solver and solve
        opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g)}
        solver = casadi.nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x']
        g = sol['g']

        # extract the optimal control and state
        sol_traj = w_opt[0:horizon * (self.n_state + self.n_control + 2 * self.n_lam)].reshape(
            (self.n_state + self.n_control + 2 * self.n_lam, -1))
        x_traj = casadi.horzcat(sol_traj[0:self.n_state, :],
                                w_opt[horizon * (self.n_state + self.n_control + 2 * self.n_lam):]).T
        u_traj = sol_traj[self.n_state:self.n_state + self.n_control, :].T
        lam_traj = sol_traj[self.n_state + self.n_control:self.n_state + self.n_control + self.n_lam, :].T

        opt_sol = {'state_traj_opt': x_traj.full(),
                   'control_traj_opt': u_traj.full(),
                   'lam_traj_opt': lam_traj.full()}

        return opt_sol

    def mpc_penalty_qp(self, init_state, horizon, mat_Q, mat_R, mat_QN, epsilon=0.001, gamma=1e-2):
        # set initial condition
        init_state = casadi.DM(init_state).full().flatten().tolist()
        Q = DM(mat_Q)
        R = DM(mat_R)
        QN = DM(mat_QN)

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = casadi.MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += init_state
        ubw += init_state
        w0 += init_state

        # formulate the NLP
        for k in range(horizon):
            # New NLP variable for the control
            Uk = casadi.MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.n_control * [-inf]
            ubw += self.n_control * [inf]
            w0 += self.n_control * [0]
            # new NLP variable for the complementarity variable
            Lamk = casadi.MX.sym('Lam_' + str(k), self.n_lam)
            w += [Lamk]
            lbw += self.n_lam * [0]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0]
            Phik = casadi.MX.sym('Phi_' + str(k), self.n_lam)
            w += [Phik]
            lbw += self.n_lam * [0]
            ubw += self.n_lam * [inf]
            w0 += self.n_lam * [0]

            # Add complementarity equation
            lcp_penalty = dot(Lamk, Phik) + \
                          0.5 / gamma * dot(self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset - Phik,
                                            self.D @ Xk + self.E @ Uk + self.F @ Lamk + self.lcp_offset - Phik)

            # Integrate till the end of the interval
            Xnext = self.A @ Xk + self.B @ Uk + self.C @ Lamk
            Ck = dot(Xk, Q @ Xk) + dot(Uk, R @ Uk) + 1 / epsilon * lcp_penalty
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = casadi.MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.n_state * [-inf]
            ubw += self.n_state * [inf]
            w0 += self.n_state * [0]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Add the final cost
        J = J + dot(Xk, QN @ Xk)

        # Create an NLP solver and solve
        opts = {'printLevel': 'none', }
        prob = {'f': J, 'x': casadi.vertcat(*w), 'g': casadi.vertcat(*g)}
        solver = qpsol('lcpSolver', 'qpoases', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x']
        g = sol['g']

        # extract the optimal control and state
        sol_traj = w_opt[0:horizon * (self.n_state + self.n_control + 2 * self.n_lam)].reshape(
            (self.n_state + self.n_control + 2 * self.n_lam, -1))
        x_traj = casadi.horzcat(sol_traj[0:self.n_state, :],
                                w_opt[horizon * (self.n_state + self.n_control + 2 * self.n_lam):]).T
        u_traj = sol_traj[self.n_state:self.n_state + self.n_control, :].T
        lam_traj = sol_traj[self.n_state + self.n_control:self.n_state + self.n_control + self.n_lam, :].T

        opt_sol = {'state_traj_opt': x_traj.full(),
                   'control_traj_opt': u_traj.full(),
                   'lam_traj_opt': lam_traj.full()}

        return opt_sol


# some facility functions
def vec2MatG(G_para, n_lam):
    if type(G_para) is casadi.SX:
        G = SX(n_lam, n_lam)
    else:
        G = DM(n_lam, n_lam)

    for j in range(n_lam):
        for i in range(j + 1):
            if i == j:
                G[i, j] = (G_para[int((j + 1) * j / 2) + i])
            else:
                G[i, j] = G_para[int((j + 1) * j / 2) + i]
    return G


# generate a random lcs
def gen_lcs(n_state, n_control, n_lam, stiffness, gb=1.0):
    A = np.random.uniform(-1, 1, size=(n_state, n_state))
    B = np.random.uniform(-1, 1, size=(n_state, n_control))
    C = 1 * np.random.uniform(-1, 1, size=(n_state, n_lam))
    dyn_offset = 1 * np.random.uniform(-1, 1, size=n_state)
    D = np.random.uniform(-1, 1, size=(n_lam, n_state))
    E = np.random.uniform(-1, 1, size=(n_lam, n_control))
    G_para = gb * np.random.uniform(-1, 1, size=int((n_lam + 1) * n_lam / 2))
    G = vec2MatG(G_para, n_lam)
    H = np.random.uniform(-1, 1, size=(n_lam, n_lam))
    F = G @ G.T + H - H.T + stiffness * np.eye(n_lam)
    lcp_offset = 1 * np.random.uniform(-1, 1, size=n_lam)

    min_sig = min(np.linalg.eigvals(F + F.T))

    lsc_theta = veccat(vec(A), vec(B),
                       vec(C),
                       dyn_offset,
                       vec(D), vec(E),
                       vec(G_para),
                       vec(H),
                       lcp_offset,
                       ).full().flatten()

    lcs_mats = {
        'n_state': n_state,
        'n_control': n_control,
        'n_lam': n_lam,
        'A': A,
        'B': B,
        'C': C,
        'D': D,
        'dyn_offset': dyn_offset,
        'E': E,
        'G_para': G_para,
        'H': H,
        'F': F,
        'lcp_offset': lcp_offset,
        'theta': lsc_theta,
        'min_sig': min_sig}

    return lcs_mats
