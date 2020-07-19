from random import sample 
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
from ranking import *
class CF:
    """docstring for CF"""
    def __init__(self, 
                Y_data, 
                Y_test, 
                k,
                true_list_for_each_user,
                hate_list_for_each_user,
                dist_func = cosine_similarity, 
                uuCF = 1, 
                num_recommend = 20, 
                ):
        '''
        initialize parameter for collaborative_filtering algorithms
        uuCF: 0 = item-item and 1 = user-user
        k: number of neighbor choosed when calculate ratings
        dist_find: distance function to  calculate similarity
        n_users, n_items: number of user and item
        self.mu: array contains the average ratings point of all user, mu[i] is mean point of user i
        Ybar_data: data after convert to sparse matrix
        num_recommend: the number of item recommended for each user
        true_list_for_each_user: true list recommend of each user in test set
        hate_list_for_each_user: the list of hated items for each user in test set
        '''

        self.uuCF = uuCF 
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.Y_test = Y_test
        self.k = k 
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
        self.num_recommend = num_recommend
        self.true_list_for_each_user = true_list_for_each_user
        self.hate_list_for_each_user = hate_list_for_each_user
    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)

    def normalize_Y(self):
        '''
        normalize the utility matrix
        '''
        # all users - first col of the Y_data
        users = self.Y_data[:, 0] 

        self.Ybar_data = self.Y_data.copy()
        # self.mu: vector of n_users number 0
        self.mu = np.zeros((self.n_users,))
        # print(self.mu)

        for n in range(self.n_users):
            # item_ids and ratings by user n
            ids = np.where(users == n)[0] #.astype(np.int32)
            item_ids = self.Y_data[ids, 1] 
            ratings = self.Y_data[ids, 2]  

            # take mean
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]
        
        # convert from ratings to sparse matrix
        # tocsr() method is to compress the original data, duplicated elements will be erased
        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()
        # print(self.Ybar.toarray())

    def similarity(self):
        eps = 1e-6
        # Ybar = utility matrix
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)
        # print(self.S)
    def refresh(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normalize_Y()
        self.similarity() 
        
    def fit(self):
        self.refresh()

    def __pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        """
        
        # find all user rated item i, find similarity between them and user u
        ids = np.where(self.Y_data[:, 1] == i)[0] #.astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        
        # find the k most similarity users, np.argsort() return indices of the sorted array(not value)
        a = np.argsort(sim)[-self.k:] 
        nearest_s = sim[a]
        
        # ratings of neighbor of user u for item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)

        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8) + self.mu[u]
    
    
    def pred(self, u, i, normalized = 1):
        """ 
        predict the rating of user u for item i (normalized)
        """
        if self.uuCF: 
            return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)

    
    def recommend_list_of_each_user(self):
        '''
        return top k recommend list for all user 
        '''
        predict_list = []
        if self.uuCF == 1:
            for i in range(self.n_users):
                for j in self.true_list_for_each_user[i]:

                    # ids = np.where(self.Y_test[:, 0] == i)[0]
                    # items_rated_by_i = self.Y_test[ids, 1].tolist()
                    recommended_items = []
                    rate_of_user_i_for_item_j = self.pred(i, j, normalized = 0)
                    recommended_items.append((rate_of_user_i_for_item_j, j))
                    
                    # choose k negative sample
                    if(len(self.hate_list_for_each_user[i]) > self.num_recommend):
                        ids = sample(range(len(self.hate_list_for_each_user[i])), self.num_recommend) 
                    else:
                        ids = range(len(self.hate_list_for_each_user[i]))
                    # print(ids)
                    for k in range(len(ids)):
                        tmp_item = self.hate_list_for_each_user[i][ids[k]]
                        rate_of_user_i_for_tmp_item = self.pred(i, tmp_item, normalized = 0)
                        recommended_items.append((rate_of_user_i_for_tmp_item, tmp_item))

                    
                    recommended_items.sort(key = lambda x : x[0], reverse=True)
                    # print(recommended_items)
                    predict_list.append([x[1] for x in recommended_items[: self.num_recommend]])
                    # print(recommended_items)    
                    # print([x[1] for x in recommended_items[: self.num_recommend]])


        
        else:
            for i in range(self.n_items):
                for j in self.true_list_for_each_user[i]:
                    recommended_items = []
                    rate_of_user_i_for_item_j = self.pred(i, j, normalized = 0)
                    recommended_items.append((rate_of_user_i_for_item_j, j))

                    # choose k negative sample
                    if(len(self.hate_list_for_each_user[i]) > self.num_recommend):
                        ids = sample(range(len(self.hate_list_for_each_user[i])), self.num_recommend) 
                    else:
                        ids = range(len(self.hate_list_for_each_user[i]))
                    # print(ids)
                    for k in range(len(ids)):
                        tmp_item = self.hate_list_for_each_user[i][ids[k]]
                        rate_of_user_i_for_tmp_item = self.pred(i, tmp_item, normalized = 0)
                        recommended_items.append((rate_of_user_i_for_tmp_item, tmp_item))

                    
                    recommended_items.sort(key = lambda x : x[0], reverse=True)
                    # print(recommended_items)
                    predict_list.append([x[1] for x in recommended_items[: self.num_recommend]])
                    # print(recommended_items)    
                    # print([x[1] for x in recommended_items[: self.num_recommend]])

        return predict_list
        
    