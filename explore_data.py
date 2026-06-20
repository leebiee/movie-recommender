import pandas as pd

movies = pd.read_csv('tmdb_5000_movies.csv')

print(movies.shape)
print(movies.columns)
print(movies.head())