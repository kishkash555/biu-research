import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import os.path as path
import pickle
import datetime
import sys
from common.fprint import fprint
now = datetime.datetime.now

def load_mnist(args):
    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)
    return train_loader, test_loader

def train(model, args, train_loader,test_loader, optimizer):
    model.train(True)
    CE = nn.CrossEntropyLoss()
    save_counter = 0
    for epoch in range(args.epochs):
        start = now()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            data = data.view([data.shape[0], -1])
            output = model(data)
            loss = CE(output, target)
            loss.backward()
            optimizer.step()
            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                fprint('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime elapsed: {}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    now()-start))
                start = now()
            if 'save_interval' in args and args.save_interval > 0 and batch_idx % args.save_interval == 0:
                fprint('Saving model {}'.format(save_counter))
                save_fname = args.data_fname.split('.')
                save_fname[0] = "{}_{:03}".format(save_fname[0], save_counter)
                save_counter += 1
                with open('.'.join(save_fname),'wb') as a:
                    pickle.dump(model.state_dict(),a)

        test(model, args, test_loader)

def test(model, args, test_loader):
    corrects, total = 0, 0
    model.train(False)
    for (data, target) in test_loader:
        data, target = data.to(args.device), target.to(args.device)
        data = data.view([-1, 784])
        output = model(data)
        y_hat = torch.argmax(output,1)
        corrects += (y_hat == target).sum()
        total += len(y_hat)
    fprint('Test: correct {} of {}, error rate {:.1%}'.format(corrects, total,  1.-float(corrects)/total ))


def mlp(input_size, output_size, hidden_sizes, args):
    hl1 = hidden_sizes[0]
    ret = nn.Sequential(
        nn.Linear(input_size,hl1),
        nn.Dropout2d(0.3,True),
        args.activation,
        nn.Linear(hl1,output_size)
        )

    return ret

    
