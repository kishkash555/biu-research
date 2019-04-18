import argparse
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

    arg = parser.parse_args()
    return arg


def parse_result_log(fo):
    layer_size_txt = fo.readline()
    layer_size = int(layer_size_txt.split(":")[-1].strip())
    print("layer_size: {}".format(layer_size))
    correct=[]
    for line in fo:
        correct.append(int(line.split(" ")[2]))
    return layer_size, correct


def load_results(dir, files):
    res = {}
    for file_name in files:
        full_fname = path.join(dir,file_name.strip())
        with open(full_fname,'rt') as a:
            layer_size, correct = parse_result_log(a)
            res[layer_size] = [1-float(c)/10000 for c in correct]
    return res

def plot_results(res):
    layer_sizes = sorted(res.keys())
    plt.figure()
    for ls in layer_sizes:
        plt.plot(res[ls],'.-')
    plt.legend(['ls'+str(l) for l in layer_sizes])

def main():
    args = arguments()
    if args.glob:
        file_list = list(glob.glob(path.join(args.dir, args.glob)))
        args.dir=''
    else:
        file_list = args.files.split(",")
    res = load_results(args.dir, file_list)
    plot_results(res)
    plt.show()
    
if __name__ == "__main__":
    main()