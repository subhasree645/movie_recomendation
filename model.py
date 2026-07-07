import pandas as pd
import numpy as np

# Load datasets
movies = pd.read_csv("dataset/tmdb_5000_movies.csv")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv")

# Display first 5 rows
print(movies.head())

# Display dataset information
print(movies.info())

# Display credits dataset
print(credits.head())
# Merge movies and credits datasets
movies = movies.merge(credits, on="title")

# Check merged dataset
print(movies.head())

# Display shape (rows, columns)
print(movies.shape)

# Select required columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Check selected columns
print(movies.head())

# Check missing values
print(movies.isnull().sum())
# Remove missing values
movies.dropna(inplace=True)

# Check shape after removing missing values
print(movies.shape)

# Check missing values again
print(movies.isnull().sum())
import ast

# Function to fetch genres and keywords
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Apply conversion
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Check output
print(movies[['title', 'genres', 'keywords']].head())
# Function to fetch top 3 cast members
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

# Apply function
movies['cast'] = movies['cast'].apply(convert3)

# Check output
print(movies[['title', 'cast']].head())
# Function to fetch director name
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Apply function
movies['crew'] = movies['crew'].apply(fetch_director)

# Check output
print(movies[['title', 'crew']].head())
# Remove spaces from names
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Check output
print(movies[['title', 'genres', 'cast', 'crew']].head())
# Convert overview into list
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create tags by combining all important features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new dataframe
new_df = movies[['movie_id', 'title', 'tags']]

# Check output
print(new_df.head())
# Convert list of tags into a single string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Check output
print(new_df.head())
from sklearn.feature_extraction.text import CountVectorizer

# Create CountVectorizer object
cv = CountVectorizer(max_features=5000, stop_words='english')

# Convert tags into vectors
vectors = cv.fit_transform(new_df['tags']).toarray()

# Check shape
print(vectors.shape)
from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between all movies
similarity = cosine_similarity(vectors)

# Check similarity matrix size
print(similarity.shape)
# Recommendation function

def recommend(movie):
    
    # Find movie index
    index = new_df[new_df['title'] == movie].index[0]

    # Get similarity scores
    distances = similarity[index]

    # Get top 5 similar movies
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x:x[1]
    )[1:6]


    print("Recommended Movies for:", movie)
    print("--------------------------------")

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# Test recommendation
recommend("Avatar")
import pickle

# Save movies dataframe
pickle.dump(new_df, open("models/movies.pkl", "wb"))

# Save similarity matrix
pickle.dump(similarity, open("models/similarity.pkl", "wb"))

print("Model saved successfully!")