# 🎬 Movie Recommender

A content-based movie recommendation system that suggests similar movies based on plot, genre, and keywords.

## How it works
- Combines each movie's overview, genres, and keywords into a single text profile
- Uses TF-IDF vectorization to convert text into numerical features
- Computes cosine similarity between all movies to find the closest matches

## Tech stack
- Python
- pandas
- scikit-learn (TF-IDF, cosine similarity)
- Streamlit (web interface)

## Dataset
[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## Run it locally
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Demo
[Add your live link here once deployed]