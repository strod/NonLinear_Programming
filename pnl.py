# Algebra and Calculus
import numpy as np

# Scientific Solvers and Models
import scipy.optimize as spo

# Timing functions
import time as tm
import inspect


def signx(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1


def sigma_norm(x, sigma=2):
    norm = ((1 / sigma) * sum([np.abs(xi) ** sigma for xi in x]))
    key = 0
    while key == 0:
        try:
            norm = sum(norm)
        except TypeError:
            key = 1
    return norm


def ghash(x):
    return np.array([max(xi, 0) for xi in x])


def penx(x, bbh=None, bbg=None, sigma=2):
    key = 0

    if bbh is None:
        hx, jhx, lx = np.array([0]), np.array([0]), np.array([0])
    else:
        hx, jhx = bbh(x)
        lx = np.array([(np.abs(xi) ** (sigma - 1)) * (signx(xi)) for xi in hx])
        key += 1

    if bbg is None:
        gx, jgx, sx = np.array([0]), np.array([0]), np.array([0])
    else:
        gx, jgx = bbg(x)
        sx = np.array([max(xi, 0) ** (sigma - 1) for xi in gx])
        key += 2

    px = sigma_norm(hx) + sigma_norm(ghash(gx))

    if key == 0:
        gpx = 0

    elif key == 1:
        gpx = np.dot(np.transpose(jhx), lx)

    elif key == 2:
        gpx = np.dot(np.transpose(jgx), sx)

    else:
        gpx = np.dot(np.transpose(jhx), lx) + np.dot(np.transpose(jgx), sx)

    return px, gpx, lx, sx


def pensolver(xo=None, ri=10, fobj=None, hx=None, gx=None, sigma=2,
              tol=0.01, method='BFGS', maxiter=100, use_grad=False,
              generate_log=False):
    """
    # PENALIZED FUNCTION SOLVER
    #
    # This function takes a given objective function with restrains and returns
    # the optimal solution as the objective function minimum
    #
    # _________________________________________________________________________
    #
    # tol : Tolerance for the approximation between the result and the feasbile
    #       set
    # ri :  Penalty factor. A constant provided as an initial guess for the
    #       penalty factor
    # fobj: Objective Function
    # px:   Penalty Function
    # hx:   Feasible Set
    # xo:   Warm Start
    #
    # RETURNS:
    #
    # Results:  Dictionary
    # rv:       Optimal Penalty Factor
    # counter:  Number of iterations until convergence
    # opt.x:    Optimal X
    """
    ini = tm.time()

    global log, opt, ttol

    px = lambda x: penx(x, bbh=hx, bbg=gx, sigma=sigma)[0]
    gpx = lambda x: penx(x, bbh=hx, bbg=gx, sigma=sigma)[1]
    lx = lambda x: penx(x, bbh=hx, bbg=gx, sigma=sigma)[2]
    sx = lambda x: penx(x, bbh=hx, bbg=gx, sigma=sigma)[3]
    obj = lambda x: fobj(x)[0]
    if use_grad:
        grad = lambda x: fobj(x)[1]
    else:
        grad = None

    results = {}
    diff = 1
    counter = 1

    if generate_log:
        right_now = tm.strftime("%Y-%m-%d_%H-%M", tm.gmtime())
        log = open("log_penalized_solver_{0}.txt".format(right_now), "w")

        log.write("LOG DOCUMENT  - PENALIZED FUNCTION SOLVER \n\n Date and Hour: {} \n\n".format(right_now))
        log.write("{0}".format(inspect.getsource(fobj)))
        log.write("{0}".format(inspect.getsource(px)))

    while (diff > tol) and (counter < maxiter):

        if generate_log:
            log.write("\n Iteration - {}".format(counter))

        nf = lambda x: obj(x) + (ri * px(x))  # Penalized Function with iterable r factor
        if use_grad:
            gradpen = lambda x: grad(x) + (ri * gpx(x))
        else:
            gradpen = None
        try:
            ttol = (2 / np.sqrt(ri)) + tol  # Temporary Tolerance
        except AttributeError:
            if ri > 2**63:
                print("R factor too big, solver is diverging.")
                print("Try with a better warm start")

                return results, ri, counter, opt.x, None, None

        opt = spo.minimize(nf, xo, method=method, tol=ttol, jac=gradpen)  # Solver
        results.update({np.round(ri, 0): (opt.x, nf(opt.x), obj(opt.x), ttol)})
        diff = px(opt.x)

        if generate_log:
            log.write("\n | r : {0:1.2f}\t\t\t |".format(ri) +
                      " Tolerance: {0:1.4f}\t |\n".format(ttol) +
                      " | Optimal Point: x = {0} \t |".format(opt.x)
                      )

            log.write("\n | Xo: {0}\t |".format(xo) +
                      " objFunc: {0:1.4f}\t | penFunc: {1:1.4f}\t diff: {2}\t\t |".format(obj(opt.x),
                                                                                          nf(opt.x),
                                                                                          diff)
                      )

            log.write("\n")

        if np.abs(diff) < tol:
            break
        if (np.abs(diff) / tol) < 2:
            ri += 1
        elif (np.abs(diff) / tol) > 2:
            ri += 20

        xo = opt.x
        counter += 1

    fin = tm.time()

    if generate_log:
        log.write("\n Total time iteration - {0:1.2f}".format(fin - ini))
        log.close()

    rv = np.round(ri, 0)
    lbd = ri*lx(opt.x)
    mi = ri*sx(opt.x)
    return results, rv, counter, opt.x, lbd, mi


def value_func(z=0, y=0, fobj=None, hx=None, gx=None, xo=None):

    if hx is None:
        new_hfunc = None
    else:
        hfunc = lambda x: hx(x)[0] - y
        jhfunc = lambda x: hx(x)[1]

        def new_hfunc(x):
            return hfunc(x), jhfunc(x)

    if gx is None:
        new_gfunc = None
    else:
        gfunc = lambda x: gx(x)[0] - z
        jgfunc = lambda x: gx(x)[1]

        def new_gfunc(x):
            return gfunc(x), jgfunc(x)

    results, rv, counter, opt_point, lbd, mi = pensolver(xo=xo, fobj=fobj, hx=new_hfunc,
                                                         gx=new_gfunc, tol=0.0001, use_grad=True)
    return fobj(opt)[0], opt_point
