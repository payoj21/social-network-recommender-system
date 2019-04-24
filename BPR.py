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
        y_pred = model._predict_user(user)
        y_true = np.zeros(n_items)
        y_true[row.indices] = 1
        auc += roc_auc_score(y_true, y_pred)
    auc /= n_users
    return auc


class DataHandler:

    def loadData(self, dataset):
        file_dir = '../data/' + dataset
        pos_file = open(file_dir+"/positive_feedback_dataframe.pkl",'rb')
        soc_file = open(file_dir+"/social_positive_feedback_dataframe.pkl",'rb')
        positive_df = pickle.load(pos_file)
        social_positive_df = pickle.load(soc_file)
        self.P = positive_df
        self.SP = social_positive_df
        self.unique_items = len(self.P['item'].unique())

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
        
    def getTrainTest(self, test_size = 0.1, seed = 20191004):
        assert test_size < 1.0 and test_size > 0.0

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
        return train, test

    def save(self, path, values, name):
        f = open(path + dataset + '/' + name + '.pkl', 'wb')
        pickle.dump(values, f)
        f.close()
        

class Sample:
    def __init__(self, P, SP, mappings):
        self.P = P
        self.SP = SP
        self.mappings = mappings
        
    def static(self, social = False):
        user = np.random.choice(self.P['user'])
        
        items = {}
        
        items['P'] = np.random.choice(self.P.query('user == @user')['item'])
        tot_items = set(self.P['item'])
        if social:
            items['SP'] = np.random.choice(self.SP.query('user == @user')['item'])
        
        pos_items = []
        for key in items:
            pos_items.append(items[key])
        
        neg_items = list(tot_items - set(pos_items))
                
        items['N'] = np.random.choice(neg_items)
        
        return user, items

class BPR:

    def __init__(self, learning_rate = 0.01, n_factors = 15, n_iters = 10, 
                 reg_u = 0.01, reg_i = 0.01, seed = 1234, verbose = True):
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.seed = seed
        self.verbose = verbose
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.batch_size = 1
        self.learning_rate = learning_rate
        
        # to avoid re-computation at predict
        self._prediction = None
        
    def fit(self, ratings, sampler):
        n_users, n_items = ratings.shape
        
        batch_size = self.batch_size
        if n_users < batch_size:
            batch_size = n_users
            sys.stderr.write('WARNING: Batch size is greater than number of users,'
                             'switching to a batch size of {}\n'.format(n_users))

        batch_iters = n_users // batch_size
        
        # initialize random weights
        rstate = np.random.RandomState(self.seed)
        self.user_factors = rstate.normal(size = (n_users, self.n_factors))
        self.item_factors = rstate.normal(size = (n_items, self.n_factors))
        
        # progress bar for training iteration if verbose is turned on
        loop = range(self.n_iters)
        if self.verbose:
            loop = trange(self.n_iters, desc = self.__class__.__name__)
        
        self.auc_scores = []
        for _ in loop:
            for _ in range(batch_iters):
                user, items = sampler.static()
                sampled_users = np.zeros(self.batch_size, dtype = np.int)
                sampled_users[0] = sampler.mappings['user_code'][user]
                
                sampled_pos_items = np.zeros(self.batch_size, dtype = np.int)
                sampled_pos_items[0] = sampler.mappings['item_code'][items['P']]
                
                '''
                sampled_soc_pos_items = np.zeros(self.batch_size, dtype = np.int)
                sampled_soc_pos_items[0] = get_mappings(mappings['item_code'], items['SP'])
                '''
                
                sampled_neg_items = np.zeros(self.batch_size, dtype = np.int)
                sampled_neg_items[0] = sampler.mappings['item_code'][items['N']]
                
                self._update(sampled_users, sampled_pos_items, sampled_neg_items)
            self.auc_scores.append(auc_score(self, ratings))
#         return self
                
    def _update(self, u, i, j):
        user_u = self.user_factors[u]
        item_i = self.item_factors[i]
        item_j = self.item_factors[j]
        
        r_uij = np.sum(user_u * (item_i - item_j), axis = 1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
        
        # repeat the 1 dimension sigmoid n_factors times so
        # the dimension will match when doing the update
        sigmoid_tiled = np.tile(sigmoid, (self.n_factors, 1)).T

        # update using gradient descent
        grad_u = sigmoid_tiled * (item_j - item_i) + self.reg_u * user_u
        grad_i = sigmoid_tiled * -user_u + self.reg_i * item_i
        grad_j = sigmoid_tiled * user_u + self.reg_i * item_j
        self.user_factors[u] -= self.learning_rate * grad_u
        self.item_factors[i] -= self.learning_rate * grad_i
        self.item_factors[j] -= self.learning_rate * grad_j
#         return self

    def predict(self):
        if self._prediction is None:
            self._prediction = self.user_factors.dot(self.item_factors.T)
            self._prediction = (self._prediction-min(self._prediction))*5/(max(self._prediction) - min(self._prediction))

        return self._prediction

    def _predict_user(self, user):
        user_pred = self.user_factors[user].dot(self.item_factors.T)
        user_pred = (user_pred-min(user_pred))*5/(max(user_pred) - min(user_pred))
        return user_pred
    
    def recommend(self, data, N = 5):
        n_users = data.shape[0]
        recommendation = np.zeros((n_users, N))
        scores = np.zeros((n_users, N))
        users = []
        ranks = []
        for user in range(n_users):
            users.append([user]*N)
            ranks.append([i for i in range(1,N+1)])
            topN_items, topN_scores = self.recommend_user(data, user, N)
            recommendation[user], scores[user] = topN_items, topN_scores

        return recommendation, scores, users, ranks        

    def recommend_user(self, data, u, N, validation = True):
        scores = self._predict_user(u)

        liked = set(data[u].indices)
        count = N + len(liked)
        if count < scores.shape[0]:
            ids = np.argpartition(scores, -count)[-count:]
            best_ids = np.argsort(scores[ids])[::-1]
            best = ids[best_ids]
            best_scores = scores[best]
        else:
            best = np.argsort(scores)[::-1]
            best_scores = scores[best]    

        topN_items = []
        topN_scores = []
        for i in range(len(best)):
            if best[i] not in liked:
                topN_items.append(best[i])
                topN_scores.append(best_scores[i])
                
        topN_items = list(islice((item for item in topN_items), N))
        topN_scores = list(islice((score for score in topN_scores), N))
        return topN_items, topN_scores

    def get_similar_items(self, N = 10, item_ids = None):
        """
        return the top N similar items for itemid, where
        cosine distance is used as the distance metric
        
        Parameters
        ----------
        N : int, default 5
            top-N similar items' N
            
        item_ids : 1d iterator, e.g. list or numpy array, default None
            the item ids that we wish to find the similar items
            of, the default None will compute the similar items
            for all the items
        
        Returns
        -------
        similar_items : 2d ndarray, shape [number of query item_ids, N]
            each row is the top-N most similar item id for each
            query item id
        """
        # cosine distance is proportional to normalized euclidean distance,
        # thus we normalize the item vectors and use euclidean metric so
        # we can use the more efficient kd-tree for nearest neighbor search;
        # also the item will always to nearest to itself, so we add 1 to 
        # get an additional nearest item and remove itself at the end
        normed_factors = normalize(self.item_factors)
        knn = NearestNeighbors(n_neighbors = N + 1, metric = 'euclidean')
        knn.fit(normed_factors)

        # returns a distance, index tuple,
        # we don't actually need the distance
        if item_ids is not None:
            normed_factors = normed_factors[item_ids]

        _, items = knn.kneighbors(normed_factors)
        similar_items = items[:, 1:].astype(np.uint32)
        return similar_items

if __name__ == '__main__':
    
    # Get the data
    dataset = str(sys.argv[1])
    dataHandler = DataHandler()
    dataHandler.loadData(dataset)
    
    X = dataHandler.getMatrix()
    X_train, X_test = dataHandler.getTrainTest()
    
    mappings = dataHandler.mappings
    
    # Create the sampler object (default: static)
    sampler = Sample(dataHandler.P, dataHandler.SP, mappings)
    
    # Initiliaze BPR params
    bpr_params = {'reg_u': 0.005,
                  'reg_i': 0.005,
                  'learning_rate': 0.3,
                  'n_iters': 100,
                  'n_factors': 10}

    # Run BPR
    bpr = BPR(**bpr_params)
    bpr.fit(X_train, sampler)
    
    def save_state(file_name):

        blob = {}

        for param in bpr.__dict__:
            blob[param] = bpr.__dict__.get(param)

        try:
            with open(file_name, 'wb') as wfile:
                pickle.dump(dict(blobs = blob), wfile, pickle.HIGHEST_PROTOCOL)
        except:
            log.error('I/O error({0}): {1}'.format(ioe.errno, ioe.strerror))
    
    save_state('../models/' + dataset + '/bpr_model.pkl')
    auc = bpr.auc_scores
    
    # Save the model and AUC scores
    
    dataHandler.save('../results/', auc, 'BPR_AUC')
    