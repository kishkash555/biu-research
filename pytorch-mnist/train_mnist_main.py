from common.fprint import init_log_file, wrapup_log_file, fprint
from train_mnist import load_mnist, train, mlp
import torch
from torch import nn
import argparse
import sys

def main():
    args = arguments()
    for hidden_layer_size in [50, 100,200,400,800,1200,1600]*10:
        data_fname = init_log_file()
        fprint('hidden layer size: {}'.format(hidden_layer_size))
        net = mlp(28*28,10,[hidden_layer_size], args).to(device=args.device)
        train_loader, test_loader = load_mnist(args)
        optimizer = torch.optim.Adam(net.parameters(),weight_decay=0.0002)
        train(net,args,train_loader,test_loader,optimizer)
        wrapup_log_file(args, net, data_fname)        

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


if __name__ == "__main__":
    sys.argv=sys.argv+ ['--epochs', '25', '--activation', 'tanh']
    main()
