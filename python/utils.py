import numpy as np


# Define frequency bands for fin whale choruses
# f0 and f1 are for the vocalization band
# nl0 and nl1 are the limits of the lower adjacent noise band
# nh0 and nh1 are the limits of the higher adjacent noise band
F20P = {'f0': 15, 'f1': 26, 'nl0': 10, 'nl1': 15, 'nh0': 30, 'nh1': 80}
LFC20 = {'f0': 17, 'f1': 25, 'nl0': 11, 'nl1': 15, 'nh0': 30, 'nh1': 33}
HFC80 = {'f0': 84, 'f1': 87, 'nl0': 81, 'nl1': 83, 'nh0': 90, 'nh1': 94}
HFC90 = {'f0': 96, 'f1': 100, 'nl0': 90, 'nl1': 94, 'nh0': 103, 'nh1': 105}


def select_fin_band(p, f, band_name):
    lims = globals()[band_name]
    band_mask = np.where((f >= lims['f0']) & (f <= lims['f1']))
    return f[band_mask], p[band_mask]


def select_noise_band(p, f, band_name):
    lims = globals()[band_name]
    band_mask = np.where((f >= lims['nl0']) & (f <= lims['nl1']) | (f >= lims['nh0']) & (f <= lims['nh1']))
    return f[band_mask], p[band_mask]


def select_total_noise_band(p, f, band_name):
    lims = globals()[band_name]
    band_mask = np.where((f >= lims['nl0']) & (f <= lims['nh1']))
    return f[band_mask], p[band_mask]


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def nextpow2(n0):
    """ Function for finding the next power of 2 """
    n = 1
    while n < n0:
        n = n * 2
    return n


def energyop(sig):
    """
    calculates the energy operator of a signal
    Method

    The Teager Energy Operator is determined as
        #(x(t)) = (dx/dt)^2+ x(t)(d^2x/dt^2) (1.1)
    in the continuous case (where x_ means the rst derivative of x, and x¨ means the second
    derivative), and as
        #[x[n]] = x^2[n] + x[n - 1]x[n + 1] (1.2)
    in the discrete case.
    Method
    Note that the function is vectorized for optimum processing speed(Keep calm and vectorize)
    Author : Hooman Sedghamiz (hoose792@student.liu.se)

    :param sig: Raw signal (Vector)
    :return: ey: Energy operator signal (ey)
             ex: Teager operator (ex)
    """

    # Operator 1
    y = np.diff(sig)
    y = np.insert(y, 0, 0)
    squ = y[1:len(y) - 1] ** 2
    oddi = y[0:len(y) - 2]
    eveni = y[2:len(y)]
    ey = squ - np.multiply(oddi, eveni)
    # [x[n]] = x^2[n] - x[n - 1]x[n + 1]
    # operator ex
    squ1 = sig[1:len(sig) - 1] ** 2
    oddi1 = sig[0:len(sig) - 2]
    eveni1 = sig[2:len(sig)]
    ex = squ1 - np.multiply(oddi1, eveni1)
    # make it the same len
    ex = np.insert(ex, 0, ex[0])
    ex = np.insert(ex, -1, ex[-1])

    return ey, ex
