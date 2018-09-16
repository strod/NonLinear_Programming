import numpy as np
import pnl
import matplotlib.pyplot as plt


def hang_chain(x1=None, y1=None, lb=np.array([4, 4, 4, 4]), nb=4,
               log=False, use_grad=True, go=None, g1=None,
               tolerance=0.00001, max_iter=100):

    """


    """
    # nk - Parametric variable to transform y coordinates in x coordinates. y[i] = x[i+nk]
    nk = nb + 1

    # Generic Initial Start
    xo = np.array([i for i in np.linspace(0, x1, nk)] +
                  [i for i in np.linspace(0, y1, nk)])

    # Knot - Instance where we keep the coordinates of border constraints
    knot = xo
    # Distance between knots (x[i], x[nk + i]) of temporary vector
    lx = lambda x: np.array([np.sqrt((x[i] - x[i - 1]) ** 2 + (x[nk + i] - x[nk + i - 1]) ** 2) for i in range(1, nk)])

    def bbh(arr):
        # h(x) - Equality Constraints
        hx = np.array([arr[0] ** 2 + arr[nk] ** 2] +
                      [lx(arr)[j] ** 2 - lb[j] ** 2 for j in range(len(lb))] +
                      [(arr[nb] - knot[nb]) ** 2 + (arr[nk + nb] - knot[nk + nb]) ** 2])

        # JACOBIAN OF h(x) - Equality Constraints
        jhx = np.array([[2 * arr[i] if i == 0 else 0 for i in range(nk)] +
                        [2 * arr[i] if i == nk else 0 for i in range(nk, nk + nk)]] +
                       [([2 * (arr[j] - arr[j - 1]) if (i == j) else
                          -2 * (arr[j] - arr[j - 1]) if (i == j - 1)
                          else 0 for i in range(nk)] +
                         [2 * (arr[j + nk] - arr[j + nk - 1]) if (i == j) else
                          -2 * (arr[j + nk] - arr[j + nk - 1]) if (i == j - 1)
                          else 0 for i in range(nk)]
                         ) for j in range(1, nk)] +
                       [[2 * (arr[i] - knot[nb]) if i == nb else 0 for i in range(nk)] +
                        [2 * (arr[i] - knot[nk + nb]) if i == nk + nb else 0 for i in range(nk, nk + nk)]])
        return hx, jhx

    if (go is None) & (g1 is None):
        bbg = None
        flag_plot = False
    elif (go is not None) & (g1 is None):
        def bbg(arr, beta=go, alfa=0, nj=nk):
            gx = np.array([beta + (arr[i] * alfa) - arr[i + nk] for i in range(nj)])
            jgx = np.array([[g1 if (i == j) else -1 if (i == j+nj) else 0 for i in range(2*nj)] for j in range(nj)])
            return gx, jgx

        flag_plot = True
    elif (go is None) & (g1 is not None):
        def bbg(arr, beta=0, alfa=g1, nj=nk):
            gx = np.array([beta + (arr[i] * alfa) - arr[i + nk] for i in range(nj)])
            jgx = np.array([[g1 if (i == j) else -1 if (i == j+nj) else 0 for i in range(2*nj)] for j in range(nj)])
            return gx, jgx

        flag_plot = True
    else:
        def bbg(arr, beta=go, alfa=g1, nj=nk):
            gx = np.array([beta + (arr[i] * alfa) - arr[i + nk] for i in range(nj)])
            jgx = np.array([[g1 if (i == j) else -1 if (i == j+nj) else 0 for i in range(2*nj)] for j in range(nj)])
            return gx, jgx

        flag_plot = True

    def bbobj(arr):
        fx = sum([(lb[i - nk - 1] / 2) * (arr[i] + arr[i - 1]) for i in range(nk + 1, nk + nb + 1)])
        grad = np.array([0 for _ in range(nk)] +
                        [lb[0] / 2] +
                        [(lb[i] + lb[i + 1]) / 2 for i in range(nb - 1)] +
                        [lb[nb - 1] / 2])
        return fx, grad

    results, rv, counter, xopt, lbd, mi = pnl.pensolver(xo, fobj=bbobj, hx=bbh, gx=bbg, tol=tolerance,
                                                        use_grad=use_grad, generate_log=log, maxiter=max_iter)

    if xopt is not None:
        plt.figure(figsize=(6, 6))
        a = xopt[:nk]
        b = xopt[nk:]
        plt.scatter(a, b)
        plt.plot(a, b, 'r--')
        if flag_plot:
            plane = lambda x: go + (g1 * x)
            w = np.linspace(0, max(xopt[:nk]), 100)
            z = np.array([plane(xi) for xi in w])
            plt.plot(w, z, 'g')
        plt.show()
    x_joints = xopt[:nk]
    y_joints = xopt[nk:]
    return x_joints, y_joints, lx(xopt)
