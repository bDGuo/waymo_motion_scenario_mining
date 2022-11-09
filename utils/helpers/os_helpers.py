import os

def mkdir(path,dirname):
    dirpath = os.path.join(path,dirname)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath
