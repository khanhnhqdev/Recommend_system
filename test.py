import Collaborative_filtering
import pandas as pd 
import numpy as np
from ranking import *
from Content_base import *
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


# ### Test ranking #####
# preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
#          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
#          []]
# # labels for the 3 users
# labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
# tmp = precision_at(preds, labels, 1)
# print(tmp) # 0.33333333333333331
# tmp = precision_at(preds, labels, 5)
# print(tmp) # 0.26666666666666666
# precision_at(preds, labels, 15)
# print(tmp) # 0.17777777777777778
# tmp = mean_average_precision(preds, labels)
# print(tmp)
# tmp = ndcg_at(preds, labels, k = 5)
# print(tmp)

import pandas as pd 
#Reading user file:
u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./Datasets/ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')

n_users = users.shape[0]
print('Number of users:', n_users)
# users.head() #uncomment this to see some few examples

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('./Datasets/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('./Datasets/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.to_numpy()
rate_test = ratings_test.to_numpy()

print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('./Datasets/ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

n_items = items.shape[0]
print('Number of items:', n_items)

X0 = items.to_numpy()
X_train_counts = X0[:, -19:]

#tfidf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

import numpy as np
def get_items_rated_by_user(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
    we care about the first three values
    return (item_ids, scores) rated by user user_id
    """
    y = rate_matrix[:,0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1 
    # while index in python starts from 0
    ids = np.where(y == user_id +1)[0] 
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0 
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)

from sklearn.linear_model import Ridge
from sklearn import linear_model

d = tfidf.shape[1] # data dimension
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

for n in range(n_users):    
    ids, scores = get_items_rated_by_user(rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept  = True)
    Xhat = tfidf[ids, :]
    
    clf.fit(Xhat, scores) 
    W[:, n] = clf.coef_
    b[0, n] = clf.intercept_

# predicted scores
Yhat = tfidf.dot(W) + b
n = 10
np.set_printoptions(precision=2) # 2 digits after . 
ids, scores = get_items_rated_by_user(rate_test, n)
Yhat[n, ids]
print('Rated movies ids :', ids )
print('True ratings     :', scores)
print('Predicted ratings:', Yhat[ids, n])
print(W[:, n])
def evaluate(Yhat, rates, W, b):
    se = 0
    cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred 
        se += (e*e).sum(axis = 0)
        cnt += e.size 
    return se/cnt

print( 'RMSE for training:', evaluate(Yhat, rate_train, W, b))
print( 'RMSE for test    :', evaluate(Yhat, rate_test, W, b))