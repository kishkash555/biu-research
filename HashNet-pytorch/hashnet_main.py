import argparse
import train_mnist
import HashNet
import torch
from os import getpid
INPUT_SIZE = 784 # 28*28 gray scale input image
OUTPUT_SIZE = 10 # number of classes 


def main():
    print("main started")
    
    args = arguments()
    args.data_fname = train_mnist.init_log_file()
    train_mnist.fprint("pid: {}".format(getpid()))
    args_str = "\n".join("{}: {}".format(k,v) for k,v in sorted(args.__dict__.items()))
    train_mnist.fprint("model initialized with command line arguments:\n{}".format(args_str))
    

    model = HashNet.HashNet2Layer(INPUT_SIZE, args.hidden, OUTPUT_SIZE, args.k1, args.k2).to(device=args.device)
    train_loader, test_loader = train_mnist.load_mnist(args)
        
    optimizer = torch.optim.Adam(model.parameters())
    train_mnist.train(model,args,train_loader, test_loader, optimizer)
    train_mnist.wrapup_log_file(args, model, args.data_fname)


def arguments():
        # Training settings
    parser = argparse.ArgumentParser(description='Hashnet MNIST')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='use a valid torch device string e.g. "cpu", "cuda:1"')
    parser.add_argument('--log-interval', type=int, default=4, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--k1', type=int, default=4900, 
                        help='how many batches to wait before logging training status')
    parser.add_argument('--k2', type=int, default=62,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--hidden', type=int, default=400,
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-interval', type=int, default=-1,
                        help='For Saving the current Model')
    arg = parser.parse_args()
    if torch.cuda.is_available():
        print('initializing {}'.format(arg.device))
        arg.device = torch.device(arg.device)        
    else:
        print('initializing cpu')
        arg.device = torch.device('cpu')
    return arg


if __name__ == "__main__":
    main()