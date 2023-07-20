from math import sqrt, pi
from numba import njit


@njit()
def distance(x1, y1, x2, y2):
    return sqrt(pow(x1-x2, 2)+pow(y1-y2, 2))


@njit()
def fix_angle(w):
    w = w % (2 * pi)
    if w > pi:
        w -= 2 * pi
    return w