import pandas as pd
import numpy as np
# from recmetrics import *
import matplotlib.pyplot as plt
from surprise import Reader, SVD, Dataset
from surprise.model_selection import train_test_split

ratings = pd.read_csv('./Datasets/ml-20m/ratings.csv')
ratings = ratings.query('rating >=3')
ratings.reset_index(drop=True, inplace=True)
print(ratings)

n=1000
users = ratings.userId.value_counts()
# print(users)
users = users[users>n].index.tolist()
# print(min(users))

ratings = ratings.query('userId in @users')
print(ratings.shape)