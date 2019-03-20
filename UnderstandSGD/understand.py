import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

class Mlp2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Mlp2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        out = self.hidden_activation(x)
        out = self.fc2(out)
        return out

    def hidden_activation(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        return out
        
def start_2h2_net(hidden_size):
    net = Mlp2(2,hidden_size, 2)
    corners = [(-1,-1),(-1,1),(1,-1),(1,1)]
    corner_results = [net(torch.FloatTensor(c)).detach().numpy() for c in corners]
    corner_diff = sorted(np.diff(cr) for cr in corner_results)

    b= (corner_diff[2] + corner_diff[1])/2
    net.fc2.bias = Parameter(torch.FloatTensor([0,-b]))
    return net

def copy_net(net):
    hidden_size = net.fc1.out_features
    new_net = Mlp2(2,hidden_size,2)
    new_net.fc1.weight = Parameter(net.fc1.weight.detach())
    new_net.fc1.bias = Parameter(net.fc1.bias.detach())
    new_net.fc2.weight = Parameter(net.fc2.weight.detach())
    new_net.fc2.bias = Parameter(net.fc2.bias.detach())
    return new_net