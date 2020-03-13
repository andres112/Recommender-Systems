

#AUTHOR: NN NN NN

import numpy as np
import pandas as pd

# read CSV
movies = pd.read_csv('data/movies.csv')


#Users ratings. 1 means like, -1 dislike, 0 not rated
john_likes=pd.DataFrame([1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0])
joan_likes=pd.DataFrame([-1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0])

#Adding the likes to the Dataset
movies["John_Likes"]=john_likes
movies["Joan_Likes"]=joan_likes

#The dataframe with the dataset
print(movies)

#Copy of the original datased  useful for Task 2
movies_weighted=movies.copy()


#************* TASK 1 ********************

#User profile by movie genre
john_likes_score=pd.DataFrame((movies.iloc[:,1:11].values*john_likes.values).sum(axis=0)).T

#**** Compute the user profile for Joan
joan_likes_score=pd.DataFrame((movies.iloc[:,1:11].values*joan_likes.values).sum(axis=0)).T #(each movies by genre * likes of de user)

#Prediction vales for  user John
pred_john=(movies.iloc[:,1:11].values*john_likes_score.values).sum(axis=1)

movies["Pred_John"]=pred_john

#Prediction vales for  user Joan
pred_joan=(movies.iloc[:,1:11].values*joan_likes_score.values).sum(axis=1) #(each genre * each genre value of user)

movies["Pred_Joan"]=pred_joan

#Showing the prediction scores for John and the names of the movies
print(movies[['Movie','John_Likes','Pred_John','Joan_Likes','Pred_Joan']])

#******************* TASK 2 ***************
#Write the code to perform the tasks described in the exercise guide for the Task 2
