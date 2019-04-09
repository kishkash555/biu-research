import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import subprocess
import os.path as path
import pickle

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

def train(model, args, train_loader,test_loader, optimizer, epochs):
    model.train(True)
    CE = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.view([-1, 784])
            output = model(data)
            loss = CE(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                fprint('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        test(model, args, test_loader)

def test(model, args, test_loader):
    corrects = total = 0
    for (data, target) in test_loader:
        data = data.view([-1, 784])
        output = model(data)
        y_hat = torch.argmax(output,1)
        corrects += (y_hat == target).sum()
        total += len(y_hat)
    fprint('Test: correct {} of {}, error rate {:.1%}'.format(corrects, total, 1-corrects/total))

def arguments():
        # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=32, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    arg = parser.parse_args()
    return arg

def mlp(input_size, output_size, hidden_sizes):
    hl1, hl2, hl3 = hidden_sizes[:3]
    ret = nn.Sequential(
        nn.Linear(input_size,hl1),
        nn.Tanh(),
        nn.Linear(hl1,hl2),
        nn.Tanh(),
        nn.Linear(hl2,hl3),
        nn.Tanh(),
        nn.Linear(hl3,output_size))

    return ret

def main():
    global log_file
    args = arguments()
    k, commit_id = pick_result_fname(qualifier='log')
    log_fname = format_filename(qualifier='log').format(commit_id, k)
    data_fname = format_filename(qualifier='data', ext='.pkl').format(commit_id, k)
   
    log_file = open(log_fname,'wt')

    net = mlp(28*28,10,[200,40,20])
    train_loader, test_loader = load_mnist(args)
    optimizer = torch.optim.Adam(net.parameters())
    train(net,args,train_loader,test_loader,optimizer,5)
    
    with open(data_fname,'wb') as a:
        pickle.dump(net.state_dict(),a)

    log_file.close()

def fprint(msg):
    global log_file
    print(msg)
    log_file.write(msg+'\n')

    
def get_commit_id():
    with open('gitlog.txt','wt') as a:
        subprocess.call('git log -1'.split(' '), stdout=a)
    with open('gitlog.txt','rt') as a:
        line = a.readline().split(' ')
        commit_id = line[1][:6]
    return commit_id


def format_filename(dir='results', qualifier='', ext='.txt'):
    return path.join(
        dir,
        '_'.join(['result'] + ([qualifier] if len(qualifier) else []) + [r'{}_{:03}'+ext])
    )

def pick_result_fname(dir='results', qualifier='',ext='.txt'):
    commit_id = get_commit_id()
    i = 0 
    output_file_tmplt = format_filename(dir, qualifier, ext)
    while path.exists(output_file_tmplt.format(commit_id,i)) and \
        path.getsize(output_file_tmplt.format(commit_id,i)):
        i += 1 
    return i, commit_id
    
if __name__ == "__main__":
    main()
