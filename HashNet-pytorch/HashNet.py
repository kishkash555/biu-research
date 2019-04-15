import xxhash as xx
import numpy as np
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import train_mnist

def make_hash(m,n,k):
    """
    create a hash function and return it
    the hash takes two input parameters, m and n, which determine the range
    allowed in the input 0..m-1, 0..n-1
    the output is in the range 0..k-1
    :param m: the possible number of the hash's first parameter
    :param n: the possible number of the hash's second parameter
    """
    my_rand_string = ''.join([chr(randint(0,255)) for _ in range(4)])

    def hf(i,j):
        if i >= m or j >= n:
            raise ValueError("check range: {} < {}, {} < {}?".format(i,m,j,n))
        num = i * n + j
        h = xx.xxh32_intdigest(my_rand_string+ int_to_str(num)) 
        return h % k
    return hf

def int_to_str(n):
    bts = ''
    while n:
        bts += chr(n % 256)
        n = n - (n % 256)
        n = int(n/256)
    return bts


class hashedLayer(nn.Module):
    def __init__(self, fan_in, fan_out, K):
        super(hashedLayer, self).__init__()
        xav = np.sqrt(6/(fan_in+fan_out))
        self.H = make_hash(fan_out, fan_in, K)

        # initilize K with the equivalent of Glorot init
        w_np = np.zeros(K,dtype=float)
        t = np.random.random((fan_in, fan_out))*2*xav- xav
        inverse_H = np.zeros((fan_out, K, fan_in),np.uint8)

        for j in range(fan_in):
            for i in range(fan_out):
                k = self.H(i,j)
                w_np[k] += t[j,i]
                inverse_H[i,k,j] = 1
        self.W = torch.tensor(w_np, dtype=torch.float32)
        self.K = K
        self.inverse_H = inverse_H
        self.fan_out = fan_out
        
    def forward(self, a):
        # first compute all possible combinations of a and K

        #mults = torch.bmm(self.K, a)
        z = []

        for i in range(self.fan_out):
            z_kj =  torch.zeros(1, requires_grad=True)
            for k in range(self.K):
                a_kj = torch.sum(a[:,self.inverse_H[i,k,:]],1,keepdim=True)  
                z_kj = z_kj + a_kj * self.W.data[k]
            z.append(z_kj)

        zz = torch.FloatTensor(z)
        print('forward: {}'.format(zz))
        return zz

class HashNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hashing_size):
        super(HashNet,self).__init__()
        
        self.fc1 = hashedLayer(input_size, hidden_size, hashing_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self,a):
        ret = nn.Tanh(self.fc1(a))
        ret = self.fc2(ret)
        return ret


def main():
    model = HashNet(28*28, 800, 10, 100)
    args = train_mnist.arguments()
    train_loader, test_loader = train_mnist.load_mnist(args)
    optimizer = torch.optim.Adam(model.parameters())
    train_mnist.train(model,args,train_loader, test_loader, optimizer, 10)

if __name__ == "__main__":
    main()