import dynet as dy
import xxhash as xx
import numpy as np
from random import randint
from collections import namedtuple


layer_namedtuple = namedtuple('layer','g,w,b,m,n,hf,type'.split(','))

def hashed_matrix(k,m,n,pc):
    """
    create a hashing dynet matrix
    """

def hashed_phi(a, hf, n):
    r"""
    creates a phi matrix from activations a using hash function hf
    each entry of phi is a sum of the corresponding a's
    $[\phi(\\mathbb{a})]_k = \sum \limits_{j:h(i,j)=k} a_j$
    
    :param a: a (dynet) vector of inputs or activations from previous layer
    :param hf: a hash function [0, len(a)] -> len(w)
    :param n: length of output vector
    :return: a dynet matrix phi which can be multiplied by w
    """
    # phi = [list() for _ in range(m)]
    m = a.dim()
    phi_i = []
    for k in range(n):
        for i in range(m):
            phi_i.append([a[j] for j in range(m) if hf(i,j)==k])
            
    phi = dy.esum(phi_i)
    return phi

def eval_network(params, train_x, train_y):
    layer_in = dy.inputTensor(train_x[np.newaxis])
    for layer in params:
        g = layer.g # the nonlinearity
        W, b = layer.w, layer.b
        m, n = layer.m, layer.n # the input and output sizes

        if layer.type == 'normal':
            layer_out = layer_in * W + b
            layer_out = g(layer_out)
        elif layer.type == 'hashed':
            layer_hf = layer.hf
            phi = hashed_phi(layer_in, layer_hf, n)
            layer_out = W * phi + b
            layer_out = g(layer_out)
        elif layer.type == 'final':
            layer_out = layer_in * W + b
            break
        layer_in = layer_out
    return layer_out

def calc_loss(params, train_x, train_y):
    layer_out = eval_network(params, train_x, train_y)
    loss = dy.pickneglogsoftmax(dy.transpose(layer_out),train_y)
    return loss, layer_out


def network1(m,n):
    """
    a 3- layer MLP of modest width
    :param m: the width of the input
    :param n: the width of the output
    """
    g = dy.tanh
    pc = dy.ParameterCollection()
    d = 10
    w1 = pc.add_parameters((m,d))
    b1 = pc.add_parameters((1,d), init=0.)
    #w2 = pc.add_parameters((d,d))
    #b2 = pc.add_parameters((1,d), init=0.)
    #w3 = pc.add_parameters((d,n))
    w3 = pc.add_parameters((d,n))
    b3 = pc.add_parameters((1,n), init=0.)
    layers = [
        layer_namedtuple(g,w1,b1,m,d,None,'normal'),
        #layer_namedtuple(g,w2,b2,d,d,None,'normal'),
        layer_namedtuple(None,w3,b3,d,n,None,'final'),
        ]
    return layers, pc

def network2(m,n,k):
    """
    a 3- layer MLP with the middle layer using hashing
    :param m: the width of the input
    :param n: the width of the output
    :param k: the number of parameters k in the hashed matrix
    """
    g = dy.tanh
    pc = dy.ParameterCollection()
    d = 50
    k = 250 # represents 90% compression
    w1 = pc.add_parameters((m,d))
    b1 = pc.add_parameters((1,m), init=0.)
    w2 = pc.add_parameters((1,k))
    b2 = pc.add_parameters((1,d), init=0.)
    w3 = pc.add_parameters((d,n))
    b3 = pc.add_parameters((1,n), init=0.)
    hf = make_hash(d,d,k)
    layers = [
        layer_namedtuple(g,w1,b1,m,d,None,'normal'),
        layer_namedtuple(g,w2,b2,d,d,hf,'hashed'),
        layer_namedtuple(None,w3,b3,d,n,None,'final'),
        ]
    return layers, pc

def train_network(train_data, dev_data, pc, params):
    
    epochs = 100
    trainer = dy.SimpleSGDTrainer(pc)
    for ep in range(epochs):
        
        i = 0
        train_loss = 0.
        train_good = 0
        # print("EPOCH {}".format(ep))
        np.random.shuffle(train_data)
        for train_x, train_y in train_data:
            dy.renew_cg()
            loss, scores = calc_loss(params, train_x, train_y)
            train_loss += loss.scalar_value()
            pred_class = scores.npvalue()
            pred_class = np.argmax(pred_class)
            train_good += pred_class == train_y
            loss.backward()
            trainer.update()
            i += 1
           
            #if i % 100 == 1:
        dev_loss, dev_acc = check_loss(dev_data, params)
        #print("epoch: {}\ttrain_loss: {:.4f}\tdev loss: {:.4f}\tacc: {:.2f}".format(i, train_loss, dev_loss, dev_acc))
        print("epoch: {}\ttrain_loss: {:.4f}\ttrain_acc: {:.2f}".format(ep, train_loss, train_good/i))
        print("epoch: {}\tdev_loss: {:.4f}\tdev_acc: {:.2f}".format(ep, dev_loss, dev_acc))
        print()

def check_loss(dev_data, params):
    cum_loss = 0.
    good = 0
    for train_x, train_y in dev_data:
        loss, score = calc_loss(params, train_x, train_y)
        cum_loss += loss.value()
        predicted_class = np.argmax(score.npvalue())
        if predicted_class == train_y:
            good += 1
    s = len(dev_data)
    return cum_loss/s, good/s



def make_hash(m,n,k):
    """
    create a hash function and return it
    the hash takes two input parameters, m and n, which determine the range
    allowed in the input 0..m-1, 0..n-1
    the output is in the range 0..k-1
    :param m: the possible number of the hash's first parameter
    :param n: the possible number of the hash's second parameter
    """
    my_rand_string = ''.join([chr(randint(0,255)) for _ in range(4)])

    def hf(i,j):
        if i >= m or j >= n:
            raise ValueError("check range: {} < {}, {} < {}?".format(i,m,j,n))
        num = i * n + j
        h = xx.xxh32_intdigest(my_rand_string+ int_to_str(num)) 
        return h % k
    
    return hf

def int_to_str(n):
    bts = ''
    while n:
        bts += chr(n % 256)
        n = n - (n % 256)
        n = int(n/256)
    return bts




