import pandas as pd
import numpy as np
import pickle
import sys

# Take the dataset from the user
dataset = str(sys.argv[1])
print ("Dataset:", dataset)
print ()
datadir = './data/' + dataset
print ("Loading data ...")
print ()

# Get paths to the datas
rating_path = datadir + '/ratings_data.txt'
trust_path = datadir + '/trust_data.txt'

# Load data into dataframes
df_ratings = pd.read_csv(rating_path, sep = ' ', header=None)
df_ratings.columns = ['user', 'item', 'rating']

df_trust = pd.read_csv(trust_path, sep = ' ', header=None)
df_trust.columns = ['user', 'friend', 'trust']

# Filter out the ratings < 4
df_observed = df_ratings.query('rating >= 4')
print ("Filtered out ratings >= 4")

# Filter out the users who have rated number of items < threshold i.e. 5
cold_start = df_observed.groupby('user').filter(lambda x : len(x) < 5)

# Save this dataframe to output file
cold_start_output = open(datadir + '/cold_start_dataframe.pkl', 'wb')
pickle.dump(cold_start, cold_start_output)
cold_start_output.close()
print ("Filtered out and saved cold start")

# Filter in the users who have rated number of items >= 5
df_observed = df_observed.groupby('user').filter(lambda x: len(x) > 5)

# Save this dataframe to output file. This is positive feedback P.
df_positive = df_observed
positive_output = open(datadir + '/positive_feedback_dataframe.pkl', 'wb')
pickle.dump(df_positive, positive_output)
print ("Filtered out and saved observed / positive (P)")

# Get unique users and items from observed data
unique_users = set(df_observed['user'].unique())
unique_items = set(df_observed['user'].unique())

# Filter out those users from trust data who do not appear in the observed dataframe
## Start with source users
source_users = set(df_trust['user'].unique())
keep_sources = unique_users & source_users
df_trust = df_trust.query('user in @keep_sources')

## Then friend users
friend_users = set(df_trust['friend'].unique())
keep_friends = unique_users & friend_users
df_trust = df_trust.query('friend in @keep_friends')

# Save this dataframe to outputfile. This is trust data intersecting with the observed data.
trust_output = open(datadir + '/trust_dataframe.pkl', 'wb')
pickle.dump(df_trust, trust_output)
trust_output.close()
print ("Filtered out and saved trust data")

# Get social positive SP(u) feedback 
soc_users = []
soc_items = []
for user in unique_users:
    # get unobserved items for user u -> nO(u)
    unobserved = unique_items - set(df_observed.query('user == @user')['item'])
    # get friends (v in V) of user u
    friends = list(df_trust.query('user == @user')['friend'])
    for friend in friends:
        # get observed items of friend v -> O(v)
        observed = set(df_observed.query('user == @friend')['item'])
        # calculate SP(u) = O(v) & nO(u)
        social_positive_feedback = list(observed & unobserved)
        # get length and append to dataframe
        n = len(social_positive_feedback)
        soc_users.extend([user]*n)
        soc_items.extend(social_positive_feedback)
        
print ("Finished getting social positive feedback")

# Convert to dataframe
social_positive_df = pd.DataFrame({'user':soc_users, 'item':soc_items})
print ("Converted to SP dataframe")

# Save this dataframe to output file. This is social positive feedback SP.
social_positive_output = open(datadir + '/social_positive_feedback_dataframe.pkl', 'wb')
pickle.dump(social_positive_df, social_positive_output)
social_positive_output.close()
print ("Saved SP to file")

# Create N(u) dataframe
neg_users = []
neg_items = []
for user in unique_users:
    # get unobserved items for user u -> nO(u)
    unobserved = unique_items - set(df_observed.query('user == @user')['item'])
    # get friends (v in V) of user u
    friends = list(df_trust.query('user == @user')['friend'])
    # initialize negative to nO(u)
    negative = unobserved
    for friend in friends:
        # get unobserved items of friend v -> nO(v)
        friend_unobserved = unique_items - set(df_observed.query('user == @friend')['item'])
        # get intersection for all friends (cumulative intersection) with nO(u)
        negative = negative & friend_unobserved
    n = len(negative)
    neg_users.extend([user]*n)
    neg_items.extend(negative)
    
print ("Finished getting negative feedback")

# Convert to dataframe
negative_df = pd.DataFrame({'user':neg_users, 'item':neg_items})
print ("Converted to N dataframe")

# Save this dataframe to output file. This is negative feedback N.
negative_output = open(datadir + '/negative_feedback_dataframe.pkl', 'wb')
pickle.dump(negative_df, negative_output)
negative_output.close()
print ("Saved N to file")

print ()
print ("... Done")