import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommender")
st.write("Pick a movie you like, and get similar recommendations.")

@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    movies = movies[['id', 'title', 'overview', 'genres', 'keywords']]

    def extract_names(text):
        try:
            items = ast.literal_eval(text)
            return ' '.join([item['name'] for item in items])
        except:
            return ''

    movies['overview'] = movies['overview'].fillna('')
    movies['genres_clean'] = movies['genres'].apply(extract_names)
    movies['keywords_clean'] = movies['keywords'].apply(extract_names)
    movies['tags'] = movies['overview'] + ' ' + movies['genres_clean'] + ' ' + movies['keywords_clean']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    similarity = cosine_similarity(tfidf_matrix)

    return movies, similarity

movies, similarity = load_data()

selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    idx = movies[movies['title'] == selected_movie].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    st.subheader("Recommended movies:")
    for i, score in scores:
        st.write(f"**{movies.iloc[i]['title']}** — similarity: {score:.2f}")