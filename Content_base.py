from random import sample 
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
from ranking import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn import linear_model

class Content_base:
    """docstring for Content_base"""
    def __init__(self, 
                Y_data, 
                Y_test, 
                true_list_for_each_user,
                hate_list_for_each_user,
                num_recommend = 20, 
                ):
        '''
        initialize parameter for algorithm
        n_users, n_items: number of user and item
        self.mu: array contains the average ratings point of all user, mu[i] is mean point of user i
        Ybar_data: data after convert to sparse matrix
        num_recommend: the number of item recommended for each user
        true_list_for_each_user: true list recommend of each user in test set
        hate_list_for_each_user: the list of hated items for each user in test set
        '''

        self.Y_data = Y_data 
        self.Y_test = Y_test
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
        self.num_recommend = num_recommend
        self.true_list_for_each_user = true_list_for_each_user
        self.hate_list_for_each_user = hate_list_for_each_user
    def build_item_profile(self):
        i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
        'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']    
        items = pd.read_csv('./Datasets/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
        items = items.to_numpy()
        items = items[:, -19:]
        transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
        self.tfidf = transformer.fit_transform(items.tolist()).toarray()
        # print(self.tfidf)
        # print(items)
                
    def fit(self):
        self.build_item_profile()
        d = self.tfidf.shape[1] # data dimension
        W = np.zeros((d, self.n_users)) # size: d x n_users
        b = np.zeros((1, self.n_users)) # size: 1 x n_users

        for n in range(self.n_users):    
            # get_items_rated_by_user n
            ids = np.where(self.Y_data[:, 0] == n)[0]
            ratings = self.Y_data[ids, 2]
            ids_of_items = self.Y_data[ids, 1]
            clf = Ridge(alpha=0.01, fit_intercept  = True)
            X = self.tfidf[ids_of_items, :]
            
            clf.fit(X, ratings) 
            W[:, n] = clf.coef_ 
            b[0, n] = clf.intercept_
            
        self.predict_ratings = self.tfidf.dot(W) + b # size: n_items x n_users
        return self.predict_ratings
    
    def recommend_list_of_each_user(self):
        '''
        return top k recommend list for all user 
        '''
        # self.fit()
        
        predict_list = []
        for i in range(self.n_users):
            for j in self.true_list_for_each_user[i]:

                # ids = np.where(self.Y_test[:, 0] == i)[0]
                # items_rated_by_i = self.Y_test[ids, 1].tolist()
                recommended_items = []
                rate_of_user_i_for_item_j = self.predict_ratings[j, i]
                recommended_items.append((rate_of_user_i_for_item_j, j))
                
                # choose k negative sample
                if(len(self.hate_list_for_each_user[i]) > self.num_recommend):
                    ids = sample(range(len(self.hate_list_for_each_user[i])), self.num_recommend) 
                else:
                    ids = range(len(self.hate_list_for_each_user[i]))
                # print(ids)
                for k in range(len(ids)):
                    tmp_item = self.hate_list_for_each_user[i][ids[k]]
                    rate_of_user_i_for_tmp_item = self.predict_ratings[tmp_item, i]
                    recommended_items.append((rate_of_user_i_for_tmp_item, tmp_item))

                
                recommended_items.sort(key = lambda x : x[0], reverse=True)
                predict_list.append([x[1] for x in recommended_items[: self.num_recommend]])



        return predict_list
        
    def pred_for_user(self, user_id):
        """
        return list of recommend for each user on unrated item of that user
        """
        ids = np.where(self.Y_data[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()              
        
        recommend_item = []
        for i in items_rated_by_u:
            rate_of_user_u_for_item_i = self.predict_ratings[i, user_id]
            recommend_item.append((rate_of_user_u_for_item_i, i))
        
        recommend_item.sort(key = lambda x : x[0], reverse=True)

        return [x[1] for x in recommend_item[: self.num_recommend]]