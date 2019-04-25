import os
import numpy as np
import random
import scipy.sparse as sparse
from scipy.sparse import *
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import pandas as pd
from math import ceil
from tqdm import trange
from sklearn.metrics import *
import sys
import pickle
from itertools import islice

def getMatrix(data):
    for col in ('item', 'user', 'rating'):
        data[col] = data[col].astype('category')

    ratings = csr_matrix((data['rating'], (data['user'].cat.codes, data['item'].cat.codes)))
    ratings.eliminate_zeros()
    print('data dimension: \n', ratings.shape)
    return ratings
        
def getTrainTest(ratings, test_size = 0.1):
    assert test_size < 1.0 and test_size > 0.0

    # Dictionary Of Keys based sparse matrix is more efficient
    # for constructing sparse matrices incrementally compared with csr_matrix
    train = ratings.copy().todok()
    test = dok_matrix(train.shape)

    for u in range(ratings.shape[0]):
        split_index = ratings[u].indices
        n_splits = ceil(test_size * split_index.shape[0])
        test_index = np.random.choice(split_index, size = n_splits, replace = False)
        test[u, test_index] = ratings[u, test_index]
        train[u, test_index] = 0

    train, test = train.tocsr(), test.tocsr()
    return train, test

def save(path, values):
    f = open(path + '.pkl', 'wb')
    pickle.dump(values, f)
    f.close()
    
dataset = sys.argv[1]
test_size = 0.1 # value from paper
folds = 10
pos_file = open("../data/" + dataset +"/positive_feedback_dataframe.pkl",'rb')
P = pickle.load(pos_file)
for i in range(folds):
    X = getMatrix(P)
    X_train, X_test = getTrainTest(X, 0.1)
    save('../data/'+ dataset + '/X_train' + str(i), X_train) 
    save('../data/'+ dataset + '/X_test' + str(i), X_test) 
