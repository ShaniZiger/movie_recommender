from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data from CSV files
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Create user-movie ratings table and compute the similarity matrix
movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
similarity_matrix = cosine_similarity(movie_matrix.T)

# Recommendation function to suggest similar movies
def recommend(movie_title, movies_df, similarity_matrix):
    try:
        movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    except IndexError:
        return ["Movie not found. Please try another one."]

    similar_movies = list(enumerate(similarity_matrix[movie_index]))
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]
    recommended_titles = [movies_df.iloc[i[0]].title for i in sorted_movies]
    return recommended_titles

# Initialize the Flask app
app = Flask(__name__)

# Home route – displays the dropdown with movie titles
@app.route('/', methods=['GET'])
def home():
    movie_titles = movies['title'].values
    return render_template('index.html', movies=movie_titles)

# Recommendation route – processes user selection and displays recommendations
@app.route('/recommend', methods=['GET'])
def recommend_movie():
    movie = request.args.get('movie')
    recommendations = recommend(movie, movies, similarity_matrix)
    return render_template('index.html', movies=movies['title'].values, recommendations=recommendations)

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
