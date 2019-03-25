from collections import Counter
from copy import deepcopy
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cosine

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim

from random import shuffle
import sys

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
        
    def trainnn(self, train_data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr = 0.1) #, lr=0.001, momentum=0.9)
        
        for ep in range(epochs):  # loop over the dataset multiple times
            print("epoch {}".format(ep))
            shuffle(train_data)
            running_loss = 0.0
            for i, data in enumerate(train_data, 0):
                # get the inputs
                inpt, label = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = self(torch.FloatTensor([inpt]))
                loss = criterion(output, torch.Tensor(data=[label]).to(dtype=torch.long))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (ep + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


def start_2h2_net(hidden_size):
    net = Mlp2(2,hidden_size, 2)
    corners = [(-1,-1),(-1,1),(1,-1),(1,1)]
    corner_results = [net(torch.FloatTensor(c)).detach().numpy() for c in corners]
    corner_diff = sorted(np.diff(cr) for cr in corner_results)

    b= (corner_diff[2] + corner_diff[1])/2
    net.fc2.bias = Parameter(torch.FloatTensor([0,-b]))
    return net


def columns_as_line_ends(m, origin=None):
    """
    represents the segments that need to be drawn to show each column in a matrix as a magnitue and direction
    :param m: the input matrix (2xK)
    :param origin: optional origin point for segments
    :returns: a dictionary with members "x" and "y", each a list of 2-tuples.
    """
    origin = origin or [0.,0.]
    x_s = m[0,:]
    y_s = m[1,:]

    ret= { 
        "x": [(origin[0], x) for x in x_s ],
        "y": [(origin[1], y) for y in y_s ],
    }
    return ret

def grad_magnitude_bins(arr2d,b,maxval):
    grad_x,grad_y = np.gradient(arr2d)
    grad_m = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    grad_m_bins = np.linspace(0,maxval,b)
    grad_m_hist = np.zeros(b, np.double)
    for i in range(b-1):
        grad_m_hist[i] = ((grad_m > grad_m_bins[i]) & (grad_m <= grad_m_bins[i+1])).sum()
    grad_m_hist[b-1] += (grad_m > maxval).sum()
    return grad_m_bins, grad_m_hist

def evaluate_network_on_mesh(net,x_s, y_s):
    coord = [np.array([x,y]) for x in x_s for y in y_s]
    net_values=net(torch.FloatTensor(coord)).detach().numpy()
    mesh_shape = (len(x_s), len(y_s))
    values_mesh0 = net_values[:,0].reshape(mesh_shape)
    values_mesh1 = net_values[:,1].reshape(mesh_shape)
    
    return values_mesh0, values_mesh1


def check_network_accuracy(net, data):
    net_values = net(torch.FloatTensor([d[0] for d in data])).detach().numpy()
    cat = np.argmax(net_values,1)
    good = sum([a==b[1] for a, b in zip(cat, data)])
    return good, len(data)

def angles(arr):
    """
    calculate the cosine of the angles between the rows
    """
    nrows = arr.shape[0]
    res = np.zeros((nrows, nrows))
    for i in range(nrows):
        for j in range(i+1):
            res[j,i] = 1-cosine(arr[i,:], arr[j,:])

    return res

if __name__ == "__main__":
    net = Mlp2(2,2,2)
    original_net = deepcopy(net)
    train_points = [np.random.uniform(-1,1,2) for _ in range(2000)]
    train_labels = [np.argmax(net(torch.FloatTensor(x)).detach().numpy()) for x in train_points]
    train_data = list(zip(train_points,train_labels)) 
    
    print(dt.datetime.now())
    print(Counter(train_labels))
 
    print("accuracy before: {}".format(check_network_accuracy(net,train_data)))
    net.trainnn(train_data,5)
    print("accuracy after: {}".format(check_network_accuracy(net,train_data)))

    x_s = y_s = np.arange(-1,1,0.05)
    b = 20 
    pre_values = evaluate_network_on_mesh(original_net,x_s, y_s)
    pre_grad_hist = grad_magnitude_bins(pre_values[0], b, 0.2)
    
    post_values = evaluate_network_on_mesh(net,x_s, y_s)
    post_grad_hist = grad_magnitude_bins(post_values[0], b, 0.2)

    
    print("\n\nLayer 1:")
    original_weights1 = original_net.fc1.weight.detach().numpy()
    trained_weights1 = net.fc1.weight.detach().numpy()
    print("original_weight_mag: {}, {}\noriginal_weight_angles:\n{}".format(
        *[norm(original_weights1[i,:]) for i in range(2)],
        angles(original_weights1)
        ))
    
    print("trained_weight_mag: {}, {}\ntrained_weight_angle:\n{}\n".format(
        *[norm(trained_weights1[i,:]) for i in range(2)],
        angles(trained_weights1)
        ))
    
    print("angles between vectors:\n{}\n".format(angles(np.concatenate([original_weights1,trained_weights1]))))


    print("\n\nLayer 2:")
    original_weights1 = original_net.fc2.weight.detach().numpy()
    trained_weights1 = net.fc2.weight.detach().numpy()
    print("original_weight_mag: {}, {}\noriginal_weight_angles:\n{}".format(
        *[norm(original_weights1[i,:]) for i in range(2)],
        angles(original_weights1)
        ))
    
    print("trained_weight_mag: {}, {}\ntrained_weight_angle:\n{}".format(
        *[norm(trained_weights1[i,:]) for i in range(2)],
        angles(trained_weights1)
        ))
    
    print("angles between vectors:\n{}".format(angles(np.concatenate([original_weights1,trained_weights1]))))
    print()
    sys.exit()

    plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.bar(*pre_grad_hist, width=0.15/b)
    plt.subplot(1,2,2)
    
    plt.bar(*post_grad_hist, width=0.15/b)
    plt.show()


import datetime as dt; 