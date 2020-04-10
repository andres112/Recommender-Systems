# Exercise 4: Item-Item Filtering
# Authors:

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import math

# Global variables
target_item = "Toy Story (1995)"
k = 5
user = "Pablo"

# **********************MAIN PROGRAM*******************************************************************
# read data
#
movie_ratings = pd.read_csv(
    'data/movies_rating.csv', header=0, index_col='Movie')

# Feel free to print the DataFrame just created
print(movie_ratings)

# Calculating the cosine similarity across items. Note that the cosine_similarity function depends from the sklearn library
# As the cosine_similarity function returns an array, we create a new data frame and assign the names to columns and rows
# which come from the original dataframe

name_movies = list(movie_ratings.index)

item_item_corr_matrix = pd.DataFrame(cosine_similarity(
    movie_ratings), columns=name_movies, index=name_movies)
print("Similarity Matrx: \n")
print(item_item_corr_matrix)

# Add the code needed to answer the requested tasks here


# Task 1: The most k similar items to the target_item ****************
if(target_item in item_item_corr_matrix.columns):
    top_items = item_item_corr_matrix[target_item].sort_values(
        ascending=False).drop(target_item)[:k]
    print(f"Top {k} similar items for {target_item}:\n", top_items)

else:
    print(f"Item {target_item} doesn't exist in dataset")

# Task 2: Prediction movies not ranked by certain user ***************
def getPrediction(item):
    # Get the already ranked items from user
    ranked_items = movie_ratings[user] != 0
    # Get the summation of the simmilar items already ranked
    sum_sim = (item_item_corr_matrix[item] * ranked_items).values.sum()
    # Sumation of the similar items by user ranking
    sim_by_rat = item_item_corr_matrix[item] * movie_ratings[user]
    sum_sim_by_rat = sim_by_rat.values.sum()

    # Return the prediction
    return sum_sim_by_rat/sum_sim

# Get the unranked movies for user
unranked_items = movie_ratings[movie_ratings[user] == 0][user]
# print(unranked_items)

# Loop for computing the prediction only of the unranked items
prediction = pd.DataFrame()
for item in unranked_items.index:
    prediction[item] = [getPrediction(item)]

print(f"\nPrediction for user {user}:\n",prediction.T.sort_values(by=0,
        ascending=False))