#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math

plt.style.use('fivethirtheight')

def fwd_diff(y, dx):
    '''
    Return forward diff approx of 1st derivative
    '''
    dydx = np.zeros(y.size)

    #Forward diff:
    dydx[:-1] = 

deltax = 0.1
x = np.arange(0, 4*np.pi, deltax)

fx = np.sin(x)
fxd1 = np.cos(x)

fig, ax = plt.subplots(1,1)
ax.plot(x, fx, alpha=.6, label='$f(x) = \sin(x)$')
ax.plot(x, fxd1, label=r'$f(x) = \frac{d\sin(x)')
