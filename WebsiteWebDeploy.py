import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd
import requests as rq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title("Movie Recommender System")
st.image("MyLogoLight.png")

# combo box with a header

header="Which Movie You are Watching"
with open("Movies.pkl","rb") as f:
    movlist= pkl.load(f)
with open("df.pkl","rb") as f:
    df = pd.DataFrame(pkl.load(f))
#with open("simi.pkl","rb") as f:
#    mov_sim=pkl.load(f)

#Using NLP for tags to find similarity
cv =CountVectorizer(max_features=5000,stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()
mov_sim=cosine_similarity(vectors)





@st.cache
def fetch_poster(movie_id):
    response= rq.get(r"https://api.themoviedb.org/3/movie/{}?api_key=4e22fb930f940bf990256cede4339c44&language=en-US".format(movie_id))
    # response is a json file
    data= response.json() # read a json in python
    return "https://image.tmdb.org/t/p/w500/"+data['poster_path']

@st.cache
def recommender(movie):
    # Find the simliarity array for movie
    mindex = df[df["title"].apply(lambda x: x.lower()) == movie.lower()].index
    if (len(mindex) < 1):
        print("This movie is not in our dataset\n Sorry for the Inconviniance....")
        return
    mindex = mindex[0]
    distance = mov_sim[mindex]

    # List such that if I sort it it not lose its index
    ind_Sim = list(enumerate(distance))  # list(index,similarity_value)
    movlist = sorted(ind_Sim, reverse=True, key=lambda x: x[1])  # Sort acc to sim value
    movlist_top10 = movlist[1:10]

    movlist_top10_ind = [i[0] for i in movlist_top10]

    top_10 = []
    top_movies_posters=[]   # Movies poster
    
    #For posters we need to sign to tmdb and get a api 
    
    for index in movlist_top10_ind:
        top_10.append(df.iloc[index].title)
        # Fetch poster from API
        top_movies_posters.append( fetch_poster(df.iloc[index].id) )

    return top_10,top_movies_posters
    # Return name posters 



movie_name_selected = st.selectbox(header,pd.unique(np.array(movlist)))

if st.button("Recommend Me Some Movies"):  # onclick it becomes true
    with st.spinner("\t\t\tLOADING........"):
        top_10_movies,top_posters= recommender(movie_name_selected)
        st.header("Recommend Movies For You are ")
        
        # Make a column such that image + text
    
        for i in range(1,10,3):
            col1,col2,col3=st.beta_columns(3)
            with col1:
                st.markdown(top_10_movies[i-1])
                st.image(top_posters[i-1])
            with col2:
                st.markdown(top_10_movies[i])
                st.image(top_posters[i])
            with col3:
                st.markdown(top_10_movies[i+1])
                st.image(top_posters[i+1])  
  
    
