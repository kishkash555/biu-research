import re
from collections import Counter, OrderedDict, defaultdict
import numpy as np
import HashNet
import pickle
from os import path
import subprocess

valid = re.compile(r'[a-zA-Z \.]')

filter_pairs = False
train_file = 'data/train'
dev_file = 'data/dev'


label_map=OrderedDict([(la, i) for i, la in enumerate('en,de,nl,it,fr,es'.split(','))])
m = 600
n = 6

def parse_lang_samples(input_stream, bigram_map = None):
    labels = []
    bigram_counters = []

    for line in input_stream:
        label, text = line.split('\t')
        labels.append(label)
        if filter_pairs:
            bigrams = [(a,b) for (a,b) in zip(text[:-1],text[1:]) if valid.match(a) and valid.match(b) ]
        else:
            bigrams = list(zip(text[:-1],text[1:]))
        bigram_counters.append(Counter(bigrams))

    total_bigram_counter = sum(bigram_counters, Counter())
    if not bigram_map:
        bigram_map = OrderedDict((x[0], i) for i, x in enumerate(total_bigram_counter.most_common(m)))

    train_y = [label_map[lbl] for lbl in labels]
    train_x = []
    for bc in bigram_counters:
        x = np.zeros(len(bigram_map))
        for b,c in bc.items():
            if b in bigram_map:
                x[bigram_map[b]] = c 
        train_x.append(x)
    
    return train_x, train_y, bigram_map

def preprocess():
    with open(train_file,'rt',encoding='utf8') as t:
        train_x, train_y, bigram_map = parse_lang_samples(t)
    
    with open(dev_file,'rt',encoding='utf8') as d:
        dev_x, dev_y, _ = parse_lang_samples(d, bigram_map)

    print("sizes: train_x {}, train_y {}, dev_x {}, dev_y {}".format(len(train_x), len(train_y), len(dev_x), len(dev_y)))
    with open('data.pickle','wb') as a:
        pickle.dump([train_x, train_y, dev_x, dev_y],a)

def run_train(output_file):
    with open('data.pickle','rb') as a:
        [train_x, train_y, dev_x, dev_y] = pickle.load(a)

    layers, pc = HashNet.network2(m,n)
    with open(output_file,'wt', encoding='utf8') as outf:
        HashNet.train_network(list(zip(train_x, train_y)),list(zip(dev_x, dev_y)), pc, layers, outf)



def get_commit_id():
    with open('gitlog.txt','wt') as a:
        subprocess.call('git log -1'.split(' '), stdout=a)
    with open('gitlog.txt','rt') as a:
        line = a.readline().split(' ')
        commit_id = line[1][:6]
    return commit_id



if __name__ == "__main__":
    if not path.exists('data.pickle'):
        preprocess()
    commit_id = get_commit_id()
    output_file_tmplt = r'result_{}_{:03}.txt'
    i = 0 
    while path.exists(output_file_tmplt.format(commit_id,i)) and \
        path.getsize(output_file_tmplt.format(commit_id,i)):
        i += 1 
    output_file = output_file_tmplt.format(commit_id,i)
    print("output will be saved in {}".format(output_file))
    run_train(output_file)

