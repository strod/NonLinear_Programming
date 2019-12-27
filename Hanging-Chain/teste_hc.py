from hanging_chain import hang_chain
import numpy as np
import pnl

bar = np.random.randint(5, 15, 10)
bbobj, bbh, xo, xopt, bars = hang_chain(x1=20, y1=10, log=True, use_grad=True, nb=10, lb=bar, go=-20, g1=-2)

fx, grad = bbobj(xo)
hx, jhx = bbh(xo)
px, gpx, lx, sx = pnl.penx(xo, bbh=bbh, bbg=None)

fx_out = bbobj(xopt)[0]
hx_out = bbh(xopt)[0]
px_out = pnl.penx(xopt, bbh=bbh, bbg=None)[0]

print("xo = {0}\n".format(xo))
print("f(x) = {0}\n".format(fx))
print("f´(x) = {0}\n".format(grad))
print("L(x) = {0}\n".format(bar))

# print("p(x) = {0}".format(px))
# print("p'(x) = {0}".format(gpx))
print("h(x) = {0}\n".format(hx))
print("h´(x) = {0}\n".format(jhx))
# print("p(x) = {0}".format(px))
# print("p´(x) = {0}".format(gpx))
# print("f´(x) + rp´(x) = {0}".format(grad + (10*gpx)))
print("\n\n x final = {0}".format(xopt))
print("\n\n f(x) final = {0}".format(fx_out))
print("\n\n L(x) final = {0}".format(bars(xopt)))
print("\n\n lbd  \n {0}".format(lx))
print("\n\n mi  \n {0}".format(sx))
# print("\n\n h(x) final = {0}".format(hx_out))
# print("\n\n p(x) final = {0}".format(px_out))
