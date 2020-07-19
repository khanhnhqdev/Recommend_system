from Collaborative_filtering import *
from Content_base import *
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from Naive_bayes import *
import sys

## Write all result to output.txt file
f = open('output.txt','w')
sys.stdout = f
### Load data for ml-20m
# ratings = pd.read_csv('./Datasets/ml-20m/ratings.csv') # sep='\t', encoding='latin-1')

# print(ratings.shape)
# num_user = int(np.max(ratings.values[:, 0]))
# num_item = int(np.max(ratings.values[:, 1]))
# print('number of user: ', num_user)
# print('number of items: ', num_item)

# rate_train, rate_test = train_test_split(ratings, test_size = 0.25, random_state = 42)
# print('train_shape: ', rate_train.shape)
# print('test_shape: ', rate_test.shape)
# rate_train = rate_train.values
# rate_test = rate_test.values

# rate_train[:, :2] -= 1
# rate_test[:, :2] -= 1
##########################

### Load data for ml-100k: in data, index start from 1, we convert index to start from 0
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
rate_train = pd.read_csv('./Datasets/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
rate_test = pd.read_csv('./Datasets/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

num_user = int(np.max(rate_train.values[:, 0]))
num_item = int(np.max(rate_train.values[:, 1]))
high = int(np.max(rate_train.values[:, 2]))
low = int(np.min(rate_train.values[:, 2]))

print('number of user: ', num_user)
print('number of items: ', num_item)
print('highest score: ', high)
print('lowest score: ', low)
rate_train = rate_train.values
rate_test = rate_test.values

print('train_shape: ', rate_train.shape)
print('test_shape: ', rate_test.shape)

rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

#########################

num_recommend_item = 20


# Get the recommend list in actual for each user, if score of user fot item > 3 that item will be recommended:
true_list_for_each_user = []
hate_list_for_each_user = []
utility_matrix_test = sparse.coo_matrix((rate_test[:, 2],
            (rate_test[:, 1], rate_test[:, 0])), (num_item, num_user))
utility_matrix_test = utility_matrix_test.tocsr()

mid = int((high + low) / 2)

for i in range(num_user):
    ids = np.where(rate_test[:, 0] == i)[0]
    items_rated_by_i = rate_test[ids, 1].tolist()     

    recommended_items = []
    hate_items = []
    for j in range(num_item):
        if j in items_rated_by_i:
            rate_of_user_i_for_item_j = utility_matrix_test[j, i]
            if rate_of_user_i_for_item_j >= 3:
                recommended_items.append(j)
            else:
                hate_items.append(j)
    # print(i, recommended_items)

    
    # print(recommended_items)
    true_list_for_each_user.append(recommended_items)    
    hate_list_for_each_user.append(hate_items)

flat_list = [item for sublist in true_list_for_each_user for item in sublist]
flat_list = [[i] for i in flat_list]
# print(true_list_for_each_user)
# print(len(flat_list))
###############################################################

## Content-base:

Content_base = Content_base(Y_data = rate_train,
                            Y_test = rate_test,
                            true_list_for_each_user = true_list_for_each_user,
                            hate_list_for_each_user = hate_list_for_each_user, 
                            num_recommend = 10
                           )
# Content_base.build_item_profile()

predict_matrix =  Content_base.fit()
n_tests = rate_test.shape[0]
SE = 0 
for n in range(n_tests):
    pred = predict_matrix[rate_test[n, 1], rate_test[n, 0]]
    # print(pred)
    SE += (pred - rate_test[n, 2])**2 

print()
print('Content base: ')
RMSE = np.sqrt(SE / n_tests)
print('RMSE: ', RMSE)

predict_list = Content_base.recommend_list_of_each_user()
print('mean_average_precision: ', mean_average_precision(predict_list, flat_list)) 

ndcg = ndcg_at(predict_list, flat_list, k = 10)
print('normalized discounted cumulative gain: ', ndcg)
print()

# ##  Collaborative filtering:
print('Collaborative filtering')
print()
# 1. User base

# Use CF user-user to train and predict

rs_user = CF(Y_data = rate_train, 
        Y_test = rate_test, 
        k = 30, 
        uuCF = 1, 
        true_list_for_each_user = true_list_for_each_user, 
        hate_list_for_each_user = hate_list_for_each_user, 
        num_recommend = 10)
rs_user.fit()
n_tests = rate_test.shape[0]
SE = 0 
for n in range(n_tests):
    pred = rs_user.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 

print('User-base: ')
RMSE = np.sqrt(SE / n_tests)
print('RMSE: ', RMSE)

predict_list = rs_user.recommend_list_of_each_user()
print('mean_average_precision: ', mean_average_precision(predict_list, flat_list)) 

ndcg = ndcg_at(predict_list, flat_list, k = 10)
print('normalized discounted cumulative gain: ', ndcg)
print()

# 2. Item base

# Use CF item-item to train and predict

rs_item = CF(Y_data = rate_train, 
        Y_test = rate_test, 
        k = 30, 
        uuCF = 0,
        true_list_for_each_user = true_list_for_each_user,
        hate_list_for_each_user = hate_list_for_each_user,
        num_recommend = 10)
rs_item.fit()
n_tests = rate_test.shape[0]
SE = 0 
for n in range(n_tests):
    pred = rs_item.pred(rate_test[n, 0], rate_test[n, 1], normalized = 0)
    SE += (pred - rate_test[n, 2])**2 

print('Item-base: ')
RMSE = np.sqrt(SE / n_tests)
print('RMSE: ', RMSE)

predict_list = rs_item.recommend_list_of_each_user()
# print(predict_list[:5])
# print(flat_list[:5])
print('mean_average_precision: ', mean_average_precision(predict_list, flat_list)) 

ndcg = ndcg_at(predict_list, flat_list, k = 10)
print('normalized discounted cumulative gain: ', ndcg)
print()

# 3. Naive bayes

Naive_bayes = Naive_bayes(Y_data = rate_train,
                            Y_test = rate_test,
                            true_list_for_each_user = true_list_for_each_user,
                            hate_list_for_each_user = hate_list_for_each_user, 
                            num_recommend = 10
                           )

Naive_bayes.fit()

n_tests = rate_test.shape[0]
SE = 0 
for n in range(n_tests):
    pred = Naive_bayes.pred(rate_test[n, 0], rate_test[n, 1])
    SE += (pred - rate_test[n, 2])**2 

print('Naive bayes base: ')
RMSE = np.sqrt(SE / n_tests)
print('RMSE: ', RMSE)

predict_list = Naive_bayes.recommend_list_of_each_user()
# print(predict_list[:5])
# print(flat_list[:5])
print('mean_average_precision: ', mean_average_precision(predict_list, flat_list)) 

ndcg = ndcg_at(predict_list, flat_list, k = 10)
print('normalized discounted cumulative gain: ', ndcg)
print()


print()
