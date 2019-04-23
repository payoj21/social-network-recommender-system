import pandas as pd

ratings = pd.read_csv('movie-ratings.txt', sep=',', header=None)

ratings = ratings.drop(columns = [2, 3, 5])

ratings.to_csv('ratings_data.txt', header=None, index=None, sep=' ', mode='w')

trust = pd.read_csv('trust.txt', sep=',', header=None)

trust.to_csv('trust_data.txt', header=None, index=None, sep=' ', mode='w')
