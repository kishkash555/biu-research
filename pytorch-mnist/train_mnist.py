import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import subprocess
import os.path as path
import pickle
import datetime
import sys
now = datetime.datetime.now

log_file = None

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
    for epoch in range(args.epochs):
        start = now()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            data = data.view([-1, 784])
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

def arguments():
        # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='use a valid torch device string e.g. "cpu", "cuda:1"')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=32, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--activation', type=str, default='tanh', metavar='N',
                        help='type of activation function, tanh/relu (default:tanh')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    arg = parser.parse_args()
    if torch.cuda.is_available():
        print('initializing {}'.format(arg.device))
        arg.device = torch.device(arg.device)        
    else:
        print('initializing cpu')
        arg.device = torch.device('cpu')
    if arg.activation == 'tanh':
        print('activation: tanh')
        arg.activation = nn.Tanh()
    else:
        print('activation: relu')
        arg.activation = nn.ReLU()
    return arg

def mlp(input_size, output_size, hidden_sizes, args):
    hl1 = hidden_sizes[0]
    ret = nn.Sequential(
        nn.Linear(input_size,hl1),
        nn.Dropout2d(0.3,True),
        args.activation,
        nn.Linear(hl1,output_size)
        )

    return ret

def main():
    global log_file
    args = arguments()
    for hidden_layer_size in [50, 100,200,400,800,1200,1600]*10:
        k, commit_id = pick_result_fname(qualifier='log')
        log_fname = format_filename(qualifier='log').format(commit_id, k)
        data_fname = format_filename(qualifier='data', ext='.pkl').format(commit_id, k)
        print('log file name: {}'.format(log_fname))
        log_file = open(log_fname,'wt')

        fprint('hidden layer size: {}'.format(hidden_layer_size))
        net = mlp(28*28,10,[hidden_layer_size], args).to(device=args.device)
        train_loader, test_loader = load_mnist(args)
        optimizer = torch.optim.Adam(net.parameters(),weight_decay=0.0002)
        train(net,args,train_loader,test_loader,optimizer)
        
        if args.save_model:
            with open(data_fname,'wb') as a:
                pickle.dump(net.state_dict(),a)

        log_file.close()

def fprint(msg):
    print(msg)
    if log_file:
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
    while path.exists(output_file_tmplt.format(commit_id,i)):
        i += 1 
    return i, commit_id
    
if __name__ == "__main__":
    sys.argv=sys.argv+ ['--epochs', '25', '--activation', 'tanh']
    main()
