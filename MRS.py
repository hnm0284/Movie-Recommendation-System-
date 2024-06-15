#!/usr/bin/env python
# coding: utf-8

# # i21-0284 i21-0293 i21-0319
# # Movie Recommendation System
# # FoSE Project

# In[73]:


import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


# # Dataset Reading

# In[74]:


path = "C:/Users/Dell"
credits_df = pd.read_csv(path + "/tmdb_5000_credits.csv")
movies_df = pd.read_csv(path + "/tmdb_5000_movies.csv")


# In[75]:


credits_df.columns = ['id','tittle','cast','crew']
movies_df = movies_df.merge(credits_df, on="id")


# In[76]:


movies_df['title'] = movies_df['title'].str.lower()
movies_df['year'] = pd.to_datetime(movies_df['release_date']).dt.year


# # Weighted Rating 

# In[77]:


# Demographic Filtering
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

# print("C: ", C)
# print("m: ", m)

new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
# print(new_movies_df.shape)


# In[78]:


def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v/(v + m) * R) + (m/(v + m) * C)


# In[79]:


new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
new_movies_df = new_movies_df.sort_values('score', ascending=False)

# new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10)


# In[80]:


# Plot top 10 movies
def plot():
    popularity = movies_df.sort_values("popularity", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 10 movies")
    plt.xlabel("Popularity")
    plt.show()
    

# plot()


# #  Content based Filtering

# In[81]:


# print(movies_df["overview"].head(5))


# In[82]:


# tfidf = TfidfVectorizer(stop_words="english")
# movies_df["overview"] = movies_df["overview"].fillna("")

# tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
# # print(tfidf_matrix.shape)


# # Compute similarity

# In[83]:


# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# # print(cosine_sim.shape)

# indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
# # print(indices.head())


# In[99]:





# In[85]:


# print("################ Content Based Filtering - plot#############")
# print()
# print("Recommendations for The Dark Knight Rises")
# print(get_recommendations("The Dark Knight Rises"))
# print()
# print("Recommendations for Avengers")
# print(get_recommendations("The Avengers"))


# In[86]:


features = ["cast", "crew", "keywords", "genres"]

for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

# movies_df[features].head(10)


# In[87]:


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


# In[88]:


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]

        if len(names) > 3:
            names = names[:3]

        return names

    return []


# In[89]:


movies_df["director"] = movies_df["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)


# In[90]:


# movies_df[['title', 'cast', 'director', 'keywords', 'genres']].head()


# In[91]:


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""


# In[92]:


features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)


# In[93]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


movies_df["soup"] = movies_df.apply(create_soup, axis=1)
# print(movies_df["soup"].head())


# In[94]:


count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])

# print(count_matrix.shape)

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
# print(cosine_sim2.shape)

movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'])


# # Recommendation Based on "Title"

# In[102]:



def get_random_element(my_list):
    # generate a random index value between 0 and len(my_list) - 1
    random_index = random.randrange(len(my_list))

    # use the random index value to access a random element in the list
    random_element = my_list[random_index]

    return random_element


def get_recommendations(title, cosine_sim=cosine_sim2):
    """
    in this function,
        we take the cosine score of given movie
        sort them based on cosine score (movie_id, cosine_score)
        take the next 10 values because the first entry is itself
        get those movie indices
        map those indices to titles
        return title list
    """

    title = title.lower()
    l = movies_df['title']
    l = list(l)
    if title not in l or title == None:
        movies = ["the prestige", "fabled", "amnesiac", "broken horses", "sinister", "2:13", "the stepfather"]
        random_number = random.randint(0, len(movies) - 1)
        return get_recommendations02(movies[random_number])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    # (a, b) where a is id of movie, b is sim_score

    movies_indices = [ind[0] for ind in sim_scores]
    movies = movies_df["title"].iloc[movies_indices]
    return (list(movies), None)


def get_recommendations02(title, cosine_sim=cosine_sim2):
    title = title.lower()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    # (a, b) where a is id of movie, b is sim_score

    movies_indices = [ind[0] for ind in sim_scores]
    movies = movies_df["title"].iloc[movies_indices]
    return (list(movies), 1)
# print("################ Content Based System - metadata #############")
# print("Recommendations for The Dark Knight Rises")
# # print(get_recommendations("The Dark Knight Rises", cosine_sim2))
# print()
# print("Recommendations for Avengers")
# # print(get_recommendations("The Avengers", cosine_sim2))


# In[104]:


# print("################ Content Based System - metadata #############")
# print("Recommendations for The Dark Knight Rises")
# print(get_recommendations("The Dark Knight Rises", cosine_sim2))
# print()
# print("Recommendations for Avengers")
# print(get_recommendations("The Avengers", cosine_sim2))


# In[105]:


########## years to search with in below function to get good results ##########
years = movies_df['year'].unique()
# years


# In[106]:


########## directors to search with in below function to get good results ##########
directors = movies_df['director'].unique()
# for i in directors:
#     print(i)


# # Recommendation Based on "Director"

# In[107]:


def get_recommendations_director(directorName):
    d = directorName
    directorName = directorName.lower()
    directorName = directorName.replace(" ", "")
    directors = movies_df['director'].unique()
    if directorName not in directors:
        print("Sorry, No data found for director ", d)
        print()
        print('But you may like: ')
        movies = ["the prestige", "fabled", "amnesiac", "broken horses", "sinister", "2:13", "the stepfather"]
        random_number = random.randint(0, len(movies) - 1)
        return get_recommendations02(movies[random_number])  # recommending movie from a random title

    grouped_by_director = movies_df.groupby('director').get_group(directorName)
    toSortAccordingTo = ['popularity', 'vote_average', 'vote_count', 'revenue', 'id']
    random_number = random.randint(0, 4)
    sorted_by_column = grouped_by_director.sort_values(toSortAccordingTo[random_number], ascending=False)
    top_five = sorted_by_column['tittle'].head(5)  # to return
    return (list(top_five), None)  # to check



# In[108]:


# get_recommendations_director('christophernolan')


# In[109]:


# get_recommendations_director('bcjs')#working on wrong input


# # Recommendation Based on "Year"

# In[110]:


def get_recommendations_year(year):
    d = year
    years = movies_df['year'].unique()
    if year not in years:
        print("Sorry, No data found for year ", d)
        print()
        print('But you may like: ')
        movies = ["the prestige", "fabled", "amnesiac", "broken horses", "sinister", "2:13", "the stepfather"]
        random_number = random.randint(0, len(movies) - 1)
        return get_recommendations02(movies[random_number])  # recommending movie of a random director

    grouped_by_year = movies_df.groupby('year').get_group(year)
    toSortAccordingTo = ['popularity', 'vote_average', 'vote_count', 'revenue', 'id']
    random_number = random.randint(0, 4)
    sorted_by_column = grouped_by_year.sort_values(toSortAccordingTo[random_number], ascending=False)
    top_five = sorted_by_column['tittle'].head(5)  # to return
    return (list(top_five), None)  # to check


# In[111]:


# get_recommendations_year(1947)


# In[112]:


# get_recommendations_year(2023)# working on wrong input


# # Recommendation Based on "Genre"

# In[113]:


genres=movies_df['genres']
genres=list(genres)
Genres=list()
for i in genres:
    for j in i:
        if j not in Genres:
            Genres.append(j)
# Genres


# In[114]:



def get_recommendations_genre(genre=str()):
    d = genre
    if genre in Genres:
        mask = movies_df['genres'].apply(lambda x: d in x)  # boolean mask to filter the DataFrame by genre
        filtered_df = movies_df[mask].sort_values('vote_average', ascending=False)
        return (list(filtered_df['title'].head(5)), None)  # return a list of the top 5 movie titles
    else:
        print("Sorry, No data found for genre ", d)
        print()
        print('But you may like: ')
        movies = ["the prestige", "fabled", "amnesiac", "broken horses", "sinister", "2:13", "the stepfather"]
        random_number = random.randint(0, len(movies) - 1)
        return get_recommendations02(movies[random_number])  # recommending movie of a random director


# In[115]:


# get_recommendations_genre('war')


# In[116]:


get_recommendations_genre('bjdis')# working on wrong input

