import pandas as pd
import ast

# Load the data
movies = pd.read_csv('tmdb_5000_movies.csv')

# Keep only the columns we need
movies = movies[['id', 'title', 'overview', 'genres', 'keywords']]

# The genres column looks like: [{"id": 28, "name": "Action"}, ...]
# We want to turn it into just: Action Adventure
def extract_names(text):
    try:
        items = ast.literal_eval(text)
        names = [item['name'] for item in items]
        return ' '.join(names)
    except:
        return ''

movies['genres_clean'] = movies['genres'].apply(extract_names)
movies['keywords_clean'] = movies['keywords'].apply(extract_names)

print(movies[['title', 'genres_clean', 'keywords_clean']].head())
# Some overviews might be missing — fill with empty string instead of NaN
movies['overview'] = movies['overview'].fillna('')

# Combine overview + genres + keywords into one text blob per movie
movies['tags'] = movies['overview'] + ' ' + movies['genres_clean'] + ' ' + movies['keywords_clean']

print(movies['tags'].iloc[0])  # show the combined tags for the first movie
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert text into numbers (TF-IDF vectors)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

print(tfidf_matrix.shape)  # rows = movies, columns = unique words

# Compute similarity between every movie and every other movie
similarity = cosine_similarity(tfidf_matrix)

print(similarity.shape)  # should be (number_of_movies, number_of_movies)
def recommend(movie_title, num_results=5):
    # Find the index of the movie that matches the title (case-insensitive)
    matches = movies[movies['title'].str.lower() == movie_title.lower()]
    
    if matches.empty:
        print(f"Movie '{movie_title}' not found in dataset.")
        return
    
    idx = matches.index[0]
    
    # Get similarity scores for this movie vs all others
    scores = list(enumerate(similarity[idx]))
    
    # Sort by similarity score, descending (skip index 0 — that's the movie itself)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_results+1]
    
    print(f"\nMovies similar to '{movie_title}':")
    for i, score in scores:
        print(f"  {movies.iloc[i]['title']}  (score: {score:.3f})")

# Test it
recommend('Avatar')
recommend('The Dark Knight Rises')