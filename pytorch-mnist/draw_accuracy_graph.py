import argparse
from collections import defaultdict
import glob
from os import path
import matplotlib.pyplot as plt

def arguments():
        # Training settings
    parser = argparse.ArgumentParser(description='parse output files')
    parser.add_argument('-f','--files', type=str, default='', metavar='FFFF',
                    help='comma-separated list of varying part of file to process')
    parser.add_argument('-d','--dir', type=str, default='.', metavar='DDD',
                    help='directory where files reside. optional/')
    parser.add_argument('-g','--glob', type=str, default='', metavar='G',
                    help='pattern for files')
    parser.add_argument('-s','--show-by',type=str, default='epoch', help='graph by epoch or layer size')
    arg = parser.parse_args()
    return arg


def parse_result_log(fo):
    layer_size_txt = fo.readline()
    layer_size = int(layer_size_txt.split(":")[-1].strip())
    print("layer_size: {}".format(layer_size))
    correct=[]
    for line in fo:
        if line.startswith('Test:'):
            correct.append(int(line.split(" ")[2]))
    return layer_size, correct


def load_results_by_epoch(dir, files):
    res = {}
    for file_name in files:
        full_fname = path.join(dir,file_name.strip())
        with open(full_fname,'rt') as a:
            layer_size, correct = parse_result_log(a)
            res[layer_size] = [1-float(c)/10000 for c in correct]
    return res

def load_results_by_layer_size(dir, files):
    res = defaultdict(list)
    for file_name in files:
        full_fname = path.join(dir,file_name.strip())
        if path.getsize(full_fname) == 0:
            continue
        with open(full_fname,'rt') as a:
            layer_size, correct = parse_result_log(a)
            res[layer_size].append( 1-float(correct[-1])/10000 )
    
    return res # {k: sorted(v) for k,v in res.items()}

def plot_results_by_epoch(res):
    layer_sizes = sorted(res.keys())
    plt.figure()
    for ls in layer_sizes:
        plt.plot(res[ls],'.-')
    plt.legend(['ls'+str(l) for l in layer_sizes])

def plot_results_by_layer_size(res):
    layer_sizes = sorted(res.keys())
    plt.figure()
    for ls in layer_sizes:
        plt.plot([ls]*len(res[ls]),res[ls],'+')
    
def main():
    args = arguments()
    if args.glob:
        file_list = list(glob.glob(path.join(args.dir, args.glob)))
        args.dir=''
    else:
        file_list = sorted(args.files.split(","))
    if args.show_by == 'epoch':
        res = load_results_by_epoch(args.dir, file_list)
        plot_results_by_epoch(res)
    else:
        res = load_results_by_layer_size(args.dir, file_list)
        plot_results_by_layer_size(res)
    plt.show()
    
if __name__ == "__main__":
    main()