from collections import defaultdict
import hashing
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

relu = torch.nn.ReLU()

def pseudo_randn(K, dtype, requires_grad):
    return None

if not torch.randn:
    torch.randn = pseudo_randn
    torch.float = float

class HashedLayerAG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, hn_hash):
        # ctx.save_for_backward(input,weight,bias)
        #ctx.hn_hash = hn_hash
        w, phi = hn_hash.expand(weight, input)

        ctx.w = torch.Tensor(w)
        ctx.phi = torch.Tensor(phi)
        output = input.mm(ctx.w.t())
        output += bias
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # input, weight, bias = ctx.saved_tensors
        w = ctx.w
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(w)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.mm(ctx.phi)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias, None

class newHashedLayer(nn.Module):
    def __init__(self, fan_in, fan_out, K):
        super().__init__()
        self.H = hashing.hnHash(fan_out, fan_in, K)
        d = fan_out*fan_in / K
        self.W = torch.nn.Parameter(torch.randn(K, dtype=torch.float, requires_grad=True)/d)
        self.bias = torch.nn.Parameter(torch.randn(fan_out, dtype=torch.float, requires_grad=True)/d)

    def forward(self, input):
        return HashedLayerAG.apply(input, self.W, self.bias, self.H)


class HashNet2Layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, k1, k2):
        super(HashNet2Layer,self).__init__()
        
        self.fc1 = newHashedLayer(input_size, hidden_size, k1) 
        self.fc2 = newHashedLayer(hidden_size, output_size, k2)
    
    def forward(self,a):
        ret = relu(self.fc1(a))
        ret = self.fc2(ret)
        return ret

