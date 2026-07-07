import streamlit as st
import pickle


# Load saved files
movies = pickle.load(open("models/movies.pkl", "rb"))
similarity = pickle.load(open("models/similarity.pkl", "rb"))


# Recommendation function
def recommend(movie):

    index = movies[movies['title'] == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x:x[1]
    )[1:6]


    recommendations = []

    for i in movie_list:
        recommendations.append(movies.iloc[i[0]].title)

    return recommendations



# Streamlit UI

st.title("🎬 Movie Recommendation System")

st.write("Find movies similar to your favourite movie")


movie_list = movies['title'].values


selected_movie = st.selectbox(
    "Choose a movie",
    movie_list
)


if st.button("Recommend"):

    results = recommend(selected_movie)

    st.subheader("Recommended Movies:")

    for movie in results:
        st.write("🎥", movie)