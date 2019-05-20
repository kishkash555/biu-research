import train_mnist
import HashNet
import torch

def main():
    print("main started")
    
    args = train_mnist.arguments()
    
    input_size = 28 * 28
    output_size = 10
    k1 = input_size * 50 # equivalent no. parameters in FC if hidden layer is 50
    k2 = 50 * output_size
    expansion_factor = 16 # ugly hack 
    hidden_size = 50 * expansion_factor 
    
    model = HashNet.HashNet2Layer(input_size, hidden_size, output_size, k1, k2).to(device=args.device)
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