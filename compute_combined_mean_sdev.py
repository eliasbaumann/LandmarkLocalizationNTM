import sys
import numpy as np

def comb_sdev(means,sdevs,n):
    sig_x = []
    sig_x2 =[]
    for m,s in zip(means,sdevs):
        sig_x.append((m*n))
        sig_x2.append(s**2 * (n-1) + ((m*n)**2/n))
    tn = len(means)*n
    tx = np.sum(sig_x)
    txx = np.sum(sig_x2)
    sd = np.sqrt((txx-tx**2/tn) / (tn-1))
    return tn, tx/tn, sd
