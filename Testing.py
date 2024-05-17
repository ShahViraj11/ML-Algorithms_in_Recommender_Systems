import pandas as pd               # For data manipulation
import numpy as np                # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import mean_squared_error        # For evaluation
from sklearn.tree import DecisionTreeRegressor        # If using Decision Trees
from sklearn.naive_bayes import GaussianNB            # If using Bayesian methods
from sklearn.neighbors import KNeighborsClassifier    # For collaborative filtering
from sklearn.feature_extraction.text import TfidfVectorizer  # For content-based filtering
import matplotlib.pyplot as plt   # For plotting

# movies = pd.read_csv('ml-25m/movies.csv')
# ratings = pd.read_csv('ml-25m/ratings.csv')

movies_df = pd.read_csv('ml-25m/movies.csv',usecols=['movieId','title'],dtype={'movieId': 'int32', 'title': 'str'})
rating_df=pd.read_csv('ml-25m/ratings.csv',usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

print(movies_df.head())
print(rating_df.head())



