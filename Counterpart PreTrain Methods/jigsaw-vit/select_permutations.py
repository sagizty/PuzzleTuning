import argparse
import itertools
import numpy as np
import random
from scipy.spatial.distance import cdist
from tqdm import tqdm
'''
Algorithm 1 from
https://arxiv.org/abs/1603.09246
'''
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_classes', type=int, default=1000, dest='n_classes')
parser.add_argument('-p', '--save_path', type=str, default='./permutations.npy', dest='save_path')
args = parser.parse_args()

n_classes = args.n_classes
P_hat = np.array(list(itertools.permutations(list(range(9)), 9)))

with tqdm(total=n_classes) as bar:
    for i in range(n_classes):
        if i == 0:
            j = random.randint(0, P_hat.shape[0])
            P =np.array(P_hat[j]).reshape([1, -1])
        else:
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        P_hat = np.delete(P_hat, j, axis=0)
        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()
        j = D.argmax()
        bar.update(1)

np.save(args.save_path, P)


