import pandas as pd
import numpy as np
from scipy import sparse

ratings = pd.read_csv('./data/ratings_data.txt', sep=' ', header=None)
ratings.columns = ['userId', 'itemId', 'rating']

user_trust = pd.read_csv('./data/trust_data.txt', sep=' ', header=None)
user_trust.columns = ['dummy', 'sourceUserId', 'targetUserId', 'trustValue']
user_trust = user_trust.drop(columns=['dummy', 'trustValue'])

def get_ratings_dataframe():
    return ratings

def get_ratings_csr():
    users = list(ratings['userId'])
    items = list(ratings['itemId'])
    user_item = np.zeros((len(users), len(items)))
    for idx, row in ratings.iterrows():
        if idx % 10000 == 0: print (idx) 
        user = row['userId']
        item = row['itemId']
        rating = row['rating']
        
        i = users.index(user)
        j = items.index(item)
        
        user_item[i][j] = rating
        
    return sparse.csr_matrix(user_item), users, items

def get_user_trust_dataframe():
    return user_trust

def get_user_trust_csr():
    sources = list(ratings['sourceUserId'])
    targets = list(ratings['targetUserId'])
    trust_matrix = np.zeros((len(sources), len(targets)))
    for idx, row in user_trust.iterrows():
        source = row['sourceUserId']
        target = row['targetUserId']
        
        i = sources.index(source)
        j = targets.index(target)
        
        trust_matrix[i][j] = 1
        
    return sparse.csr_matrix(trust_matrix), sources, targets