from collections import defaultdict
import numpy as np
from random import randint
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import train_mnist
import xxhash as xx

relu = torch.nn.ReLU()

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
        fan_in += 1 # for bias term
        super(hashedLayer, self).__init__()
        self.H = make_hash(fan_out, fan_in, K)
        hh=defaultdict(set)

        for j in range(fan_in):
            for i in range(fan_out):
                k = self.H(i,j)
                hh[(i,k)].add(j)


        d = fan_out*fan_in / K
        self.W = torch.nn.Parameter(torch.randn(K, dtype=torch.float, requires_grad=True)/d)
        self.K = K
        self.fan_out = fan_out
        self.hh = hh

        
    def forward(self, a):
        b = torch.ones(a.shape[0],1)
        if a.is_cuda:
            get_cuda_device = a.get_device()
            b.to(get_cuda_device)
        a = torch.cat([a,b], dim =1)
        a_kj = torch.zeros(self.fan_out, a.shape[0], self.K)
        for k in range(self.K):
            for i in range(self.fan_out):
               a_kj[i,:,k] = sum(a[:,j] for j in self.hh[i,k]) 
        zz = torch.matmul(a_kj, self.W)
        return zz.t()

class HashNet2Layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, k1, k2):
        super(HashNet2Layer,self).__init__()
        
        self.fc1 = hashedLayer(input_size, hidden_size, k1) # +1 in order to include a constant "1" (the bias term)
        self.fc2 = hashedLayer(hidden_size, output_size, k2)
#       self.fc2 = nn.Linear(hidden_size, output_size), 
    
    def forward(self,a):
        ret = relu(self.fc1(a))
        ret = self.fc2(ret)
        return ret


def main():
    print("main started")
    
    args = train_mnist.arguments()
    
    input_size = 28 * 28
    output_size = 10
    k1 = input_size * 50 # equivalent no. parameters in FC if hidden layer is 50
    k2 = 50 * output_size
    expansion_factor = args.seed # ugly hack 
    hidden_size = 50 * expansion_factor 
    
    model = HashNet2Layer(input_size, hidden_size, output_size, k1, k2).to(device=args.device)
    train_loader, test_loader = train_mnist.load_mnist(args)
    

    data_file = train_mnist.init_log_file()
    train_mnist.fprint("model initialized with expansion factor: {}".format(expansion_factor))
    
    optimizer = torch.optim.Adam(model.parameters())
    train_mnist.train(model,args,train_loader, test_loader, optimizer)
    train_mnist.wrapup_log_file(args, model, data_file)

def mock_train_loader():
    train_loader = [(
    torch.randn(64,10), 
    torch.tensor(np.random.rand(64)*10, dtype= torch.long)
    ) for _ in range(3)]
    return train_loader

if __name__ == "__main__":
    main()