import pickle
import sys
from matplotlib import pyplot as plt
from BPR import BPR
from SBPR1 import SBPR1
from SBPR2 import SBPR2
from lenskit import topn
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
from itertools import islice

def saveAUCPlot(dataset, aname):
    f = open('./results/' + dataset + '/' + aname + '_AUC.pkl', 'rb')
    auc = pickle.load(f)
    f.close()
    plt.plot(range(len(auc)), auc)
    plt.savefig('./results/' + dataset + '/' + aname + '_100epochs.png')
    
def loadModel(dataset, aname):
    if aname == 'BPR':
        f = open('./models/' + dataset + '/bpr_model.pkl', 'rb')
        model = BPR()
        bpr_params = pickle.load(f)['blobs']
        model.user_factors = bpr_params['user_factors']
        model.item_factors = bpr_params['item_factors']
        f.close()
    elif aname == 'SBPR1':
        f = open('./models/' + dataset + '/sbpr1_model.pkl', 'rb')
        model = SBPR1()
        sbpr1_params = pickle.load(f)['blobs']
        model.user_factors = sbpr1_params['user_factors']
        model.item_factors = sbpr1_params['item_factors']
        f.close()
    elif aname == 'SBPR2':
        f = open('./models/' + dataset + '/sbpr2_model.pkl', 'rb')
        model = SBPR2()
        sbpr2_params = pickle.load(f)['blobs']
        model.user_factors = sbpr2_params['user_factors']
        model.item_factors = sbpr2_params['item_factors']
        f.close()
        
    return model, df  

def getMatrix(data):
    for col in ('item', 'user', 'rating'):
        data[col] = data[col].astype('category')

    code_user = dict(zip(data['user'].cat.codes, data['user']))
    user_code = dict(zip(data['user'], data['user'].cat.codes))
    code_item = dict(zip(data['item'].cat.codes, data['item']))
    item_code = dict(zip(data['item'], data['item'].cat.codes))

    mappings = {'code_user' : code_user, 'user_code' : user_code, 'code_item' : code_item, 'item_code' : item_code}

    ratings = csr_matrix((data['rating'], (data['user'].cat.codes, data['item'].cat.codes)))
    ratings.eliminate_zeros()
    print('data dimension: \n', ratings.shape)
    return ratings, mappings
        
def getTrainTest(ratings, test_size = 0.1, seed = 20191004):
    assert test_size < 1.0 and test_size > 0.0

    # Dictionary Of Keys based sparse matrix is more efficient
    # for constructing sparse matrices incrementally compared with csr_matrix
    train = ratings.copy().todok()
    test = dok_matrix(train.shape)

    rstate = np.random.RandomState(seed)
    for u in range(ratings.shape[0]):
        split_index = ratings[u].indices
        n_splits = ceil(test_size * split_index.shape[0])
        test_index = rstate.choice(split_index, size = n_splits, replace = False)
        test[u, test_index] = ratings[u, test_index]
        train[u, test_index] = 0

    train, test = train.tocsr(), test.tocsr()
    return train, test


def getRecommendations(model, X_test, N, df, mappings):
    recommendation, scores, users, ranks = model.recommend(X_test, N = N)
    flatten = lambda l: [item for sublist in l for item in sublist]
    df_test = pd.DataFrame({'user': flatten(users), 'item': flatten(recommendation)})
    df_test['item'] = [mappings['code_item'][item] for item in df_test['item'].astype(int)]
    df_test['user'] = [mappings['code_user'][user] for user in df_test['user'].astype(int)]
    df_test['score'] = flatten(scores)
    df_test['rank'] = flatten(ranks)
    return df_test

def getMetrics(df_test, df, N):
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    rla.add_metric(topn.recall)
    results = rla.compute(df_test, df)
    ndcg_mean = results['ndcg'].mean()
    recall_mean = results['recall'].mean()
    return ndcg_mean, recall_mean

def coldStart(mappings, df, df_test, X_test):
    total_cold_start_users = df.groupby('user').filter(lambda x : len(x) <= 6)['user'].unique()
    coded_user = [mappings['user_code'][user] for user in total_cold_start_users]
    true_ratings = dict([user, mappings['code_item'][X_test[mappings['user_code'][user]].indices[0]]] for user in total_cold_start_users)
    cold_start_ratings = pd.DataFrame(columns=['user','item','rating'])
    for user in total_cold_start_users:
        cold_start_ratings = pd.concat([cold_start_ratings, df.query('user == @user & item == @true_ratings[@user]')], axis = 0)
    cold_start_users = list(cold_start_ratings['user'].unique())
    df_cold_start = df_test.query('user in @cold_start_users')
    return df_cold_start

dataset = sys.argv[1]
file_dir = '../data/' + dataset
pos_file = open(file_dir+"/positive_feedback_dataframe.pkl",'rb')
df = pickle.load(pos_file)

X, mappings = getMatrix(df)
_, X_test = getTrainTest(X)
print ("Got matrices")

# test_users = []
# test_items = []
# test_ratings = []
# for i in range(X_test.shape[0]):
#     y = X_test.getrow(i).indices
#     if len(y) > 0:
#         data = X_test.getrow(i).data
#         user = mappings['code_user'][i]
#         item = [mappings['code_item'][y_i] for y_i in y]
#         test_users.extend([user]*len(y))
#         test_items.extend(item)
#         test_ratings.extend(data)
# 
# test_data = pd.DataFrame({'user': test_users, 'item': test_items, 'rating': test_ratings})

for aname in ['BPR', 'SBPR1', 'SBPR2']:
    print (aname)
    print ('--------------------------------')
    try:
        f = open('./results/' + dataset + '/' + aname + '_AUC.pkl', 'rb')
        auc = pickle.load(f)
        f.close()
        plt.plot(range(len(auc)), auc)
        plt.savefig('./results/' + dataset + '/' + aname + '_100epochs.png')
        print ("Saved AUC")
        model, df = loadModel(dataset, aname)
        
        df_test = getRecommendations(model, X_test, 5, df, mappings)
        df_cold_start = coldStart(mappings, df, df_test, X_test)
        ndcg_at_5, recall_at_5 = getMetrics(df_test, df, 5)
        print ("NDCG@5 =", ndcg_at_5)
        print ("Recall@5 =", recall_at_5)
        print ()
        print ("--- Cold Start")
        ndcg_at_5, recall_at_5 = getMetrics(df_cold_start, df, 5)
        print ("NDCG@5 =", ndcg_at_5)
        print ("Recall@5 =", recall_at_5)
        print ()
        df_test = getRecommendations(model, X_test, 10, df, mappings)
        df_cold_start = coldStart(mappings, df, df_test, X_test)
        ndcg_at_10, recall_at_10 = getMetrics(df_test, df, 10)
        print ("NDCG@10 =", ndcg_at_10)
        print ("Recall@10 =", recall_at_10)
        print ()
        print ("--- Cold Start")
        ndcg_at_10, recall_at_10 = getMetrics(df_cold_start, df, 10)
        print ("NDCG@5 =", ndcg_at_10)
        print ("Recall@5 =", recall_at_10)   
        
        print ()
    except:
        print ('exception')
        pass