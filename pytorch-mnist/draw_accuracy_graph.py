import argparse
from collections import defaultdict
import glob
from os import path
import matplotlib.pyplot as plt
import pandas as pd
import sys

def arguments():
        # Training settings
    parser = argparse.ArgumentParser(description='parse output files')
    parser.add_argument('-f','--files', type=str, default='', metavar='FFFF',
                    help='comma-separated list of varying part of file to process')
    parser.add_argument('-d','--dir', type=str, default='.', metavar='DDD',
                    help='directory where files reside. optional/')
    parser.add_argument('-g','--glob', type=str, default='', metavar='G',
                    help='pattern for files')
    parser.add_argument('-s','--show-by',type=str, default='default', help='graph by epoch or layer size')
    parser.add_argument('-p', '--pandas', action='store_true', help="don't graph, return a pandas DataFrame")
    arg = parser.parse_args()
    return arg


def parse_result_log(fo):
    layer_size_txt = fo.readline()
    layer_size = int(layer_size_txt.split(":")[-1].strip())
    correct=[]
    for line in fo:
        if line.startswith('Test:'):
            correct.append(int(line.split(" ")[2]))
    return layer_size, correct


def load_results_by_epoch(dir, files, output_pandas = False):
    res = {}
    res_pd = []
    layer_size_format='{:03}_{}'.format
    i=0
    for file_name in files: 
        full_fname = path.join(dir,file_name.strip())
        if path.getsize(full_fname) == 0:
            continue
        with open(full_fname,'rt') as a:
            layer_size, correct = parse_result_log(a)
            key = layer_size_format(i,layer_size)
            res[key] = [1-float(c)/10000 for c in correct]
            if output_pandas:
                res_pd.append({
                    'file': full_fname, 
                    'layer_size': layer_size,
                    'layer_size_format': key,
                    'epochs': len(correct),
                    'final_error': res[key][-1],
                    'error_list': res[key]
                 })
        i += 1
    if output_pandas:
        return pd.DataFrame(data=res_pd, columns='file,layer_size,layer_size_format,epochs,final_error,error_list'.split(','))
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
    plt.legend(layer_sizes)

def plot_results_by_layer_size(res):
    layer_sizes = sorted(res.keys())
    plt.figure()
    for ls in layer_sizes:
        plt.plot([ls]*len(res[ls]),res[ls],'+')
    
def main(command_line_args = None):
    if command_line_args:
        sys.argv = [sys.argv[0]] + command_line_args
    args = arguments()
    if args.pandas and args.show_by != 'default':
        raise ValueError('cannot use pandas and show-by together')
    elif args.show_by == 'default':
        args.show_by = 'epoch'
    if args.glob:
        file_list = list(glob.glob(path.join(args.dir, args.glob)))
        print('found {} files'.format(len(file_list)    ))
        args.dir=''
    else:
        file_list = sorted(args.files.split(","))
    if args.pandas:
        res = load_results_by_epoch(args.dir, file_list, True)
        return res
    elif args.show_by == 'epoch':
        res = load_results_by_epoch(args.dir, file_list)
        plot_results_by_epoch(res)
    else:
        res = load_results_by_layer_size(args.dir, file_list)
        plot_results_by_layer_size(res)
    plt.show()
    
if __name__ == "__main__":
    main()