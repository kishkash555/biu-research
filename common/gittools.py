import subprocess

def get_commit_id():
    with open('gitlog.txt','wt') as a:
        subprocess.call('git log -1'.split(' '), stdout=a)
    with open('gitlog.txt','rt') as a:
        line = a.readline().split(' ')
        commit_id = line[1][:6]
    return commit_id

