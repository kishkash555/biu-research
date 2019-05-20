import numpy as np
from random import randint
import xxhash as xx

class hnHash:
    def __init__(self,m,n,k):
        """
        create a hash function and return it
        the hash takes two input parameters, m and n, which determine the range
        allowed in the input 0..m-1, 0..n-1
        the output is in the range 0..k-1
        :param m: the possible number of the hash's first parameter
        :param n: the possible number of the hash's second parameter
        """
        self.rand_string = ''.join([chr(randint(0,255)) for _ in range(4)])
        self.mn = m,n
        self.K = k

    def hf(self, i,j):
        m, n = self.mn
        if i >= m or j >= n:
            raise ValueError("check range: {} < {}, {} < {}?".format(i,m,j,n))
        num = i * n + j
        h = xx.xxh32_intdigest(self.rand_string+ int_to_str(num)) 
        return h % self.K
    
    def expand(self, weight, input):
        m,n = self.mn
        w = np.zeros((m, n))
        phi = np.zeros((m, self.K))
        for i in range(m):
            for j in range(n):
                k = self.hf(i,j)
                w[i,j] = weight[k]
                phi[i,k] = phi[i,k] + input[:,j].sum().item()
        return w, phi
                
def int_to_str(n):
    bts = ''
    while n:
        bts += chr(n % 256)
        n = n - (n % 256)
        n = int(n/256)
    return bts
