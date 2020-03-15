

# AUTHOR: Maria Alejandra Cardona; Andres Felipe Dorado

import numpy as np
import pandas as pd
import math

def bestPrediction(movies, user):
    no_rated = movies.loc[movies['{}_Likes'.format(user)] == 0]
    max_prediction = no_rated.loc[movies['Pred_{}'.format(user)] == no_rated['Pred_{}'.format(user)].max()]
    print ("\n** The most like prediction for {}: \n".format(user),max_prediction[['Movie', '{}_Likes'.format(user), 'Pred_{}'.format(user)]])

def worstPrediction(movies, user):
    no_rated = movies.loc[movies['{}_Likes'.format(user)] == 0]
    worst_prediction = no_rated.loc[movies['Pred_{}'.format(user)] < 0]
    print ("\n** Dislikes prediction for {}: \n".format(user),worst_prediction[['Movie', '{}_Likes'.format(user), 'Pred_{}'.format(user)]])


# read CSV
movies = pd.read_csv('data/movies.csv')


# Users ratings. 1 means like, -1 dislike, 0 not rated
john_likes = pd.DataFrame(
    [1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0])
joan_likes = pd.DataFrame(
    [-1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0])

# Adding the likes to the Dataset
movies["John_Likes"] = john_likes
movies["Joan_Likes"] = joan_likes

# The dataframe with the dataset
print(movies, '\n')

# Copy of the original datased  useful for Task 2
movies_weighted = movies.copy()


# ************* TASK 1 ********************

# User profile by movie genre
john_likes_score = pd.DataFrame(
    (movies.iloc[:, 1:11].values*john_likes.values).sum(axis=0)).T

# **** Compute the user profile for Joan
# (each movies by genre * likes of de user)
joan_likes_score = pd.DataFrame(
    (movies.iloc[:, 1:11].values*joan_likes.values).sum(axis=0)).T

# Prediction vales for  user John
pred_john = (movies.iloc[:, 1:11].values*john_likes_score.values).sum(axis=1)

movies["Pred_John"] = pred_john

# Prediction vales for  user Joan
# (each genre * each genre value of user)
pred_joan = (movies.iloc[:, 1:11].values*joan_likes_score.values).sum(axis=1)

movies["Pred_Joan"] = pred_joan

# Showing the prediction scores for John and the names of the movies
print(movies[['Movie', 'John_Likes', 'Pred_John', 'Joan_Likes', 'Pred_Joan']], '\n')

bestPrediction(movies,"John")
bestPrediction(movies,"Joan")

worstPrediction(movies,"John")
worstPrediction(movies,"Joan")

# ******************* TASK 2 ***************
# Write the code to perform the tasks described in the exercise guide for the Task 2

print("\n***************************************************************\n")

total_sqrt = pd.DataFrame(
    map((lambda x: math.sqrt(x)), movies_weighted.iloc[:, 11:12].values)).T

movie_values = pd.DataFrame(data=movies_weighted.iloc[:, 1:11].values).T
movie_sqrt_list = movie_values.div(total_sqrt.iloc[0]).T

# User profile by movie genre
john_likes_score = pd.DataFrame(
    (movie_sqrt_list*john_likes.values).sum(axis=0)).T

# **** Compute the user profile for Joan
# (each movies by genre * likes of de user)
joan_likes_score = pd.DataFrame(
    (movie_sqrt_list*joan_likes.values).sum(axis=0)).T

# Prediction vales for  user John
pred_john = (movie_sqrt_list*john_likes_score.values).sum(axis=1)

movies_weighted["Pred_John"] = pred_john

# Prediction vales for  user Joan
# (each genre * each genre value of user)
pred_joan = (movie_sqrt_list*joan_likes_score.values).sum(axis=1)

movies_weighted["Pred_Joan"] = pred_joan

# Showing the prediction scores for John and the names of the movies
print(movies_weighted[['Movie', 'John_Likes', 'Pred_John', 'Joan_Likes', 'Pred_Joan']])

bestPrediction(movies_weighted,"John")
bestPrediction(movies_weighted,"Joan")

worstPrediction(movies_weighted,"John")
worstPrediction(movies_weighted,"Joan")
