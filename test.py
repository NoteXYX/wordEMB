import numpy as np
import argparse
import collections
import sys
from embeddings import *

BATCH_SIZE = 500

def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k