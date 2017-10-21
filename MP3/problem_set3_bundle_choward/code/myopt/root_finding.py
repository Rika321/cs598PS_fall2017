import numpy as np

def bisection(func, bounds, eps = 1e-3, maxIter = 100):
    (lb,ub) = bounds
    values  = [0,0,0]
    iter    = 0
    values[0] = func(lb)
    values[2] = func(ub)

    # check that the initial function values bound a 0
    if values[0]*values[2] > 0 :
        raise ValueError('The initial function values do not appear to bound a 0 location.')

    # loop until criteria is satisfied
    while abs(values[2]-values[0]) > eps and iter < maxIter :
        iter += 1
        center = 0.5*(lb + ub)
        values[1] = func(center)
        if values[1]*values[2] > 0:
            ub = center
            values[2] = values[1]
        elif values[1] == 0:
            return (center,iter)
        else:
            lb = center
            values[0] = values[1]

    # return output center
    return (0.5*(lb+ub),iter)


def ibisection(func, bounds, maxIter=100):
    (lb, ub)    = bounds
    values      = [0, 0, 0]
    iter        = 0
    values[0]   = func(lb)
    values[2]   = func(ub)
    bb          = lb
    bv          = 1e100

    # update best value and best cost based on current values computed
    if abs(values[0]) < bv:
        bv = abs(values[0])
        bb = lb
    if abs(values[2]) < bv:
        bv = abs(values[2])
        bb = ub

    # check that the initial function values bound a 0
    if values[0] * values[2] > 0:
        raise ValueError('The initial function values do not appear to bound a 0 location. The values are v(L) = {0} and v(R) = {1}'.format(values[0],values[2]))

    # loop until criteria is satisfied
    while (ub-lb) > 1 and iter < maxIter:
        iter        += 1
        center      = int((lb + ub)/2)
        values[1]   = func(center)

        if abs(values[1]) < bv:
            bv = abs(values[1])
            bb = center

        if values[1] * values[2] > 0:
            ub          = center
            values[2]   = values[1]
        elif values[1] == 0:
            return (center,iter)
        else:
            lb          = center
            values[0]   = values[1]

    # return output center
    return (bb,iter)