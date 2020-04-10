# Exercise 3: User-User Filtering
# Authors: Maria Alejandra Cardozo & Andres Dorado

import numpy as np
import pandas as pd
import math

user = "Pedro"
k = 2

# SupportFunctions

# This function receives the correlation values and a list of ratings, if  the rating is different than Nan
# the we multiply  the value of the correlations by the rating otherwise no. The sum of all the operations is returned.


def sumproduct(correlations, ratings, users_mean=np.zeros(k)):
    sum = 0
    for i in range(len(list(correlations))):
        if not math.isnan(ratings[i]):
            sum = sum+(correlations[i]*(ratings[i] - users_mean[i]))
    return sum

# This function receives the correlation values and a list of ratings, if  the rating is different than Nan
# the we sum up the value of the correlations, otherwise no. The value of the sum is returned


def sumif(correlations, ratings):
    sum = 0
    for i in range(len(list(correlations))):
        if not math.isnan(ratings[i]):
            sum = sum+correlations[i]
    return sum

# Implementation of the Not Normalized score rating function


def prediction_not_normalized(correlations, ratings):
    # k is the neigborhood size
    if sumif(correlations.values.reshape(k), ratings) > 0:
        return sumproduct(correlations.values.reshape(k), ratings) / sumif(correlations.values.reshape(k), ratings)
    return np.nan

# Implementation of the Normalized score rating function


def prediction_normalized(correlations, ratings, users_mean, ru_mean):
    # k is the neigborhood size
    if sumif(correlations.values.reshape(k), ratings) > 0:
        return ru_mean + (sumproduct(correlations.values.reshape(k), ratings, users_mean) / sumif(correlations.values.reshape(k), ratings))
    return np.nan

# get the N-top movies for user based on the predictions


def getTopN(movies, user, n, previousRanked=False):
    if (not previousRanked):
        top_movies = movies[movies[user].isnull()].sort_values(
            by='Prediction', ascending=False)[:n]
    else:
        top_movies = movies.sort_values(
            by='Prediction', ascending=False)[:n]
    return top_movies


# **********************MAIN PROGRAM*******************************************************************
# read data
movie_ratings = pd.read_csv('data/movie-ratings.csv', header=0)
# copying datased for task 2
movie_ratings_weighted = movie_ratings.copy()

# Feel free to print the DataFrame just created
# print(movie_ratings)

# Calculating the pearson correlation among users
user_by_user_corr_matrix = movie_ratings.corr(method='pearson')

# Print the correlation Matrix
# print(user_by_user_corr_matrix)

# Show correlation values for user Sofia and sorted from highest to lowest
# print(user_by_user_corr_matrix.loc[user].sort_values(ascending=False))

# The Top5 similar users for user Sofia, the highest the correlation value, the more similar. Observe that the dataframe
# has been sliced from the index 1, since in the index 0 the value will be 1.00 (self-correlation)
# To check the similar users of other users just change the label, e.g., user_by_user_corr_matrix.loc['Arielle']
corr_top5 = (user_by_user_corr_matrix.loc[user].sort_values(
    ascending=False)[1:(k+1)]).to_frame().T
print(corr_top5)
print("\n")


# This list is basically to select the rating of the top5 users
selection_labels = ['Movie'] + corr_top5.columns.tolist()
# print(selection_labels)
# Here we are using the previous list to select the ratings
rating_top5 = movie_ratings.loc[:, selection_labels]
# print(rating_top5)

# Empty list to store the results of the prediction scores
prediction_results = []

# We iterate over the rows of the top5 similar users ratings
for index, row in rating_top5.iterrows():
    # Getting the rating values for the movies of each user of the top 5
    ratings_row = row[selection_labels[1:]].values
    # Computing the prediction values, using the not normalized model. We call this function sending as parameters
    # the correlation values of the top 5 users and the ratings they have assigned to the items
    pred_value = prediction_not_normalized(corr_top5, ratings_row)
    # List with the results of the prediction, we add a new result in each iteration, one per each item
    prediction_results.append(pred_value)

# Adding a new column to our original DataFrame
movie_ratings['Prediction'] = prediction_results

# Visualizing the items sorted from highest prediction score to lowest
# We should recommend the items with the highest prediction score
# print(movie_ratings.sort_values(by='Prediction', ascending=False))

print("*** Top movies for {} without normalization***\n".format(user))
# please use True or False in the third parameter to include or not the movies prior rated by user target
print(getTopN(movie_ratings, user, 3, True))

# ****************************** Normalized Implementation**************************

#We get the average of all the movie ratings for each user
user_mean=movie_ratings_weighted.mean(axis = 0, skipna = True)
# Print the user ratings mean value
# print(user_mean)

#Taking the average ratings only for the top 5 neighbors of Arielle 
mean_top5=[]
for namen in corr_top5.columns:    
    mean_top5.append(user_mean[namen])

#Taking the average rating for the target user (Arielle)
ru_mean=user_mean[user]

prediction_results = []

# We iterate over the rows of the top5 similar users ratings
for index, row in rating_top5.iterrows():
    # Getting the rating values for the movies of each user of the top 5
    ratings_row = row[selection_labels[1:]].values
    # Computing the prediction values, using the not normalized model. We call this function sending as parameters
    # the correlation values of the top 5 users and the ratings they have assigned to the items
    pred_value = prediction_normalized(
        corr_top5, ratings_row, mean_top5, ru_mean)
    # List with the results of the prediction, we add a new result in each iteration, one per each item
    prediction_results.append(pred_value)

# Adding a new column to our original DataFrame
movie_ratings_weighted['Prediction'] = prediction_results

# Print the predictions using normalized method
# print(movie_ratings_weighted)

print("*** Top movies for {} with normalization***\n".format(user))
# please use True or False in the third parameter to include or not the movies prior rated by user target
print(getTopN(movie_ratings_weighted, user, 3, True))
