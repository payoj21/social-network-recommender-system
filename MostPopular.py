import os
import numpy as np
import random
import scipy.sparse as sparse
from scipy.sparse import *
from sklearn.utils import shuffle
import pandas as pd
from math import ceil
from tqdm import trange
from sklearn.metrics import *
import sys
import pickle
from itertools import islice

def auc_score(model, ratings):
    auc = 0.0
    n_users, n_items = ratings.shape
    for user, row in enumerate(ratings):
        y_pred = np.zeros(n_items)
        y_pred[model.predict()] = 1
        y_true = np.zeros(n_items)
        y_true[row.indices] = 1
        auc += roc_auc_score(y_true, y_pred)
    auc /= n_users
    return auc


class DataHandler:

    def loadData(self, dataset):
        file_dir = '../data/' + dataset
        pos_file = open(file_dir+"/positive_feedback_dataframe.pkl",'rb')
        positive_df = pickle.load(pos_file)
        self.P = positive_df
        self.unique_items = len(self.P['item'].unique())
        self.topN = list(self.P.groupby('item').size().sort_values(ascending=False).reset_index()['item'])

    def getMatrix(self):
        data = self.P
        for col in ('item', 'user', 'rating'):
            data[col] = data[col].astype('category')

        code_user = dict(zip(data['user'].cat.codes, data['user']))
        user_code = dict(zip(data['user'], data['user'].cat.codes))
        code_item = dict(zip(data['item'].cat.codes, data['item']))
        item_code = dict(zip(data['item'], data['item'].cat.codes))

        self.mappings = {'code_user' : code_user, 'user_code' : user_code, 'code_item' : code_item, 'item_code' : item_code}

        self.ratings = csr_matrix((data['rating'], (data['user'].cat.codes, data['item'].cat.codes)))
        self.ratings.eliminate_zeros()
        print('data dimension: \n', self.ratings.shape)
        return self.ratings
        
    def getTrainTest(self, test_size = 0.1, seed = 20191004, fold=None, dataset=None):
        assert test_size < 1.0 and test_size > 0.0
        if fold is None:
            self.ratings = self.getMatrix()
            # Dictionary Of Keys based sparse matrix is more efficient
            # for constructing sparse matrices incrementally compared with csr_matrix
            train = self.ratings.copy().todok()
            test = dok_matrix(train.shape)

            rstate = np.random.RandomState(seed)
            for u in range(self.ratings.shape[0]):
                split_index = self.ratings[u].indices
                n_splits = ceil(test_size * split_index.shape[0])
                test_index = rstate.choice(split_index, size = n_splits, replace = False)
                test[u, test_index] = self.ratings[u, test_index]
                train[u, test_index] = 0

            train, test = train.tocsr(), test.tocsr()
        else:
            f = open('../data/' + dataset + '/X_train' + str(fold) + '.pkl', 'rb')
            train = pickle.load(f)
            f.close()
            f = open('../data/' + dataset + '/X_test' + str(fold) + '.pkl', 'rb')
            test = pickle.load(f)
        return train, test

    def save(self, path, values, name):
        f = open(path + dataset + '/' + name + '.pkl', 'wb')
        pickle.dump(values, f)
        f.close()

class MostPopular:
        
    def fit(self, ratings, mappings, topN):
        
        n_users, n_items = ratings.shape
        self.mappings = mappings
        self.topN = topN
        
        return self

    def predict(self):
        indices = [self.mappings['item_code'][i] for i in self.topN[:500]]
        return np.random.choice(indices)
    
    def recommend(self, data, N = 5):
        n_users = data.shape[0]
        recommendation = np.zeros((n_users, N))
        users = np.zeros((n_users, N))
        ranks = np.zeros((n_users, N))
        for user in range(n_users):
            u = []
            r = []   
            topN_items = []
            for i in range(N):
                u.append(user)
                r.append(i+1)
                topN_items.append(self.predict())
            users[user] = u
            ranks[user] = r
            recommendation[user] = topN_items

        return recommendation, users, ranks

if __name__ == '__main__':
    
    # Get the data
    dataset = str(sys.argv[1])
    dataHandler = DataHandler()
    dataHandler.loadData(dataset)
    
#     X = dataHandler.getMatrix()
    X_train, X_test = dataHandler.getTrainTest() # change folds here for crossvalidation
    
    mappings = dataHandler.mappings

    # Run MostPopular
    mp = MostPopular()
    mp = mp.fit(X_train, mappings, dataHandler.topN)
    
    def save_state(file_name):

        blob = {}

        for param in mp.__dict__:
            blob[param] = mp.__dict__.get(param)

        with open(file_name, 'wb') as wfile:
            pickle.dump(dict(blobs = blob), wfile, pickle.HIGHEST_PROTOCOL)

    
    save_state('./models/' + dataset + '/mp_model.pkl')
    auc = auc_score(mp, X_test)
    
    # Save the model and AUC scores
    
    dataHandler.save('./results/', auc, 'MostPopular_AUC')
    
