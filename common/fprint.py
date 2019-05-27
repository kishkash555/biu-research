import pickle
from os import path
log_file = None
from common.gittools import get_commit_id

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


def init_log_file():
    global log_file
    k, commit_id = pick_result_fname(qualifier='log')
    log_fname = format_filename(qualifier='log').format(commit_id, k)
    data_fname = format_filename(qualifier='data', ext='.pkl').format(commit_id, k)
    print('log file name: {}'.format(log_fname))
    log_file = open(log_fname,'wt')
    return data_fname

def wrapup_log_file(args, net, data_fname):
    if args.save_model:
        with open(data_fname,'wb') as a:
            pickle.dump(net.state_dict(),a)
    log_file.close()


def fprint(msg):
    print(msg, flush=True)
    if log_file:
        log_file.write(msg+'\n')
        log_file.flush()
