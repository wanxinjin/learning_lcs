from casadi import *
import numpy as np


# solve the LCP problem
def SolveLCP_IPOPT(M, q, algorithm='ipopt'):
    if q.ndim == 1:
        n_lam = q.size
        batch_size = 1
        q = DM(q)
        vecM = vec(M)

    else:
        n_lam = q.shape[0]
        batch_size = q.shape[1]
        q = DM(q)
        vecM = np.tile(vec(M), (1, batch_size))

    batch_p = np.vstack((vecM, q))

    lam = SX.sym('lam', n_lam)
    F = SX.sym('F', n_lam, n_lam)
    c = SX.sym('c', n_lam)
    f = dot(lam, F @ lam + c)
    g = F @ lam + c
    p = vertcat(vec(F), vec(c))
    opts = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
    prob = {'f': f, 'x': lam, 'g': g, 'p': p}
    lcp_solver = casadi.nlpsol('lcp_solver', 'ipopt', prob, opts)

    sol = lcp_solver(lbg=0, lbx=0.0, p=batch_p)
    lam_batch = sol['x'].full()
    f_batch = sol['f'].full()
    g_batch = sol['g'].full()

    if batch_size == 1:
        lam_batch = lam_batch.flatten()

    return lam_batch


''' LCPSolve(M,q): procedure to solve the linear complementarity problem:
       w = M z + q
       w and z >= 0
       w'z = 0
   The procedure takes the matrix M and vector q as arguments.  The
   procedure has three returns.  The first and second returns are
   the final values of the vectors w and z found by complementary
   pivoting.  The third return is a 2 by 1 vector.  Its first
   component is a 1 if the algorithm was successful, and a 2 if a
   ray termination resulted.  The second component is the value of
   the artificial variable upon termination of the algorithm.
   The third component is the number of iterations performed in the
   outer loop.

   Derived from: http://www1.american.edu/academic.depts/cas/econ/gaussres/optimize/quadprog.src
   (original GAUSS learning_lcs by Rob Dittmar <dittmar@stls.frb.org> )
   Lemke's Complementary Pivot algorithm is used here. For a description, see:
   http://ioe.engin.umich.edu/people/fac/books/murty/linear_complementarity_webbook/kat2.pdf
Copyright (c) 2010 Rob Dittmar, Enzo Michelangeli and IT Vision Ltd
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''


def LCPSolve_Lemke(M, q, pivtol=1e-8):  # pivtol = smallest allowable pivot element
    rayTerm = False
    loopcount = 0
    if (q >= 0.).all():  # Test missing in Rob Dittmar's learning_lcs
        # As w - Mz = q, if q >= 0 then w = q and z = 0
        w = q
        z = np.zeros_like(q)
        retcode = 0.
    else:
        dimen = M.shape[0]  # number of rows
        # Create initial tableau
        tableau = np.hstack([np.eye(dimen), -M, -np.ones((dimen, 1)), np.asarray(np.asmatrix(q).T)])
        # Let artificial variable enter the basis
        basis = np.arange(dimen)  # basis contains a set of COLUMN indices in the tableau
        locat = np.argmin(tableau[:, 2 * dimen + 1])  # row of minimum element in column 2*dimen+1 (last of tableau)
        basis[locat] = 2 * dimen  # replace that choice with the row
        cand = locat + dimen
        pivot = tableau[locat, :] / tableau[locat, 2 * dimen]
        tableau -= tableau[:,
                   2 * dimen:2 * dimen + 1] * pivot  # from each column subtract the column 2*dimen, multiplied by pivot
        tableau[locat, :] = pivot  # set all elements of row locat to pivot
        # Perform complementary pivoting
        oldDivideErr = np.seterr(divide='ignore')[
            'divide']  # suppress warnings or exceptions on zerodivide inside numpy
        while np.amax(basis) == 2 * dimen:
            loopcount += 1
            eMs = tableau[:, cand]  # Note: eMs is a view, not a copy! Do not assign to it...
            missmask = eMs <= 0.
            quots = tableau[:, 2 * dimen + 1] / eMs  # sometimes eMs elements are zero, but we suppressed warnings...
            quots[missmask] = np.Inf  # in any event, we set to +Inf elements of quots corresp. to eMs <= 0.
            locat = np.argmin(quots)
            if abs(eMs[locat]) > pivtol and not missmask.all():  # and if at least one element is not missing
                # reduce tableau
                pivot = tableau[locat, :] / tableau[locat, cand]
                tableau -= tableau[:, cand:cand + 1] * pivot
                tableau[locat, :] = pivot
                oldVar = basis[locat]
                # New variable enters the basis
                basis[locat] = cand
                # Select next candidate for entering the basis
                if oldVar >= dimen:
                    cand = oldVar - dimen
                else:
                    cand = oldVar + dimen
            else:
                rayTerm = True
                break
        np.seterr(divide=oldDivideErr)  # restore original handling of zerodivide in Numpy
        # Return solution to LCP
        vars = np.zeros(2 * dimen + 1)
        vars[basis] = tableau[:, 2 * dimen + 1]
        w = vars[:dimen]
        z = vars[dimen:2 * dimen]
        retcode = vars[2 * dimen]
    # end if (q >= 0.).all()

    if rayTerm:
        retcode = (2, retcode, loopcount)  # ray termination
    else:
        retcode = (1, retcode, loopcount)  # success
    return (w, z, retcode)


def SolveLCP_Lemke(M, q):
    if q.ndim > 1:
        batch_size = q.shape[1]
        print(batch_size)
        lam_batch = np.zeros_like(q)
        g_batch = np.zeros_like(q)
        flag_batch = []
        for i in range(batch_size):
            g, lam, flag = LCPSolve_Lemke(M, q[:, i])
            lam_batch[:, i] = lam
            g_batch[:, i] = g
            flag_batch += [flag]

    else:
        g_batch, lam_batch, flag_batch = LCPSolve_Lemke(M, q)

    print('g non-negativeness:', (g_batch >= -0e-8).all())
    print('lam non-negativeness:', (lam_batch >= -0e-8).all())
    print('g_batch*lam_batch:', (abs(g_batch * lam_batch) < 1e-8).all())

    # breakpoint()

    return lam_batch


# solve the
def SolveLCP_CVX(M, q):
    M = DM(M)
    q = DM(q)

    n_lam = M.shape[0]
    lam = SX.sym('lam', n_lam)
    f = dot(lam, M @ lam + q)
    g = M @ lam + q

    quadprog = {'x': lam, 'f': f, 'g': g}
    opts = {'printLevel': 'none', }
    lcpSolver = qpsol('lcpSolver', 'qpoases', quadprog, opts)
    sol = lcpSolver(lbx=0., lbg=0.)
    lam_sol = sol['x']

    return lam_sol.full().flatten(), (M@lam_sol+q).full().flatten()


# solve the
def SolveLCP_CVX2(M, q):
    M = DM(M)
    q = DM(q)

    n_lam = M.shape[0]
    lam = SX.sym('lam', n_lam)
    f = 0.5 * dot(lam, M @ lam) + dot(lam, q)

    quadprog = {'x': lam, 'f': f,}
    opts = {'printLevel': 'none', }
    lcpSolver = qpsol('lcpSolver', 'qpoases', quadprog, opts)
    sol = lcpSolver(lbx=0.)
    lam_sol = sol['x']

    return lam_sol.full().flatten(), (M@lam_sol+q).full().flatten()
