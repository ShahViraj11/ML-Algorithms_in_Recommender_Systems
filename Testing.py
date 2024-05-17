# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load data
movies_df = pd.read_csv('ml-25m/movies.csv', usecols=['movieId', 'title', 'genres'], dtype={'movieId': 'int32', 'title': 'str', 'genres': 'str'})
rating_df = pd.read_csv('ml-25m/ratings.csv', usecols=['userId', 'movieId', 'rating'], dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

# Merge and aggregate ratings
df = pd.merge(rating_df, movies_df, on='movieId')
movie_ratings = df.groupby('movieId').rating.mean()  # Average rating per movie

# Prepare features from genres
tfidf = TfidfVectorizer(stop_words='english')
movie_genres = tfidf.fit_transform(movies_df.set_index('movieId').loc[movie_ratings.index]['genres'])  # Ensure alignment

# Split data
X_train, X_test, y_train, y_test = train_test_split(movie_genres, movie_ratings, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
rmse = root_mean_squared_error(y_test, predictions)  # Use root_mean_squared_error
print(f'RMSE: {rmse}')
