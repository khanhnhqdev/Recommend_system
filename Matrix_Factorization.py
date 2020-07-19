from random import sample 
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 

class MF:
    def __init__(self, 
                Y_data, 
                K, 
                true_list_for_each_user,
                hate_list_for_each_user,
                lam = 0.1, 
                Xinit = None, 
                Winit = None, 
                learning_rate = 0.01,
                max_iter = 1000, 
                print_every = 100, 
                user_based = 1, 
                num_recommend = 10):
        '''
        initialize parameter for algorithm
        n_users, n_items: number of user and item
        lam: regularization parameter
        learning_rate: learning rate for gradient descent
        self.mu: array contains the average ratings point of all user, mu[i] is mean point of user i
        num_recommend: the number of item recommended for each user
        true_list_for_each_user: true list recommend of each user in test set
        hate_list_for_each_user: the list of hated items for each user in test set
        print_every: print results after print_every iterations
        user_base: user_base or item_base
        K: number of latent features
        '''
        self.Y_raw_data = Y_data
        self.K = K
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.user_based = user_based
        self.n_users = int(np.max(Y_data[:, 0])) + 1 
        self.n_items = int(np.max(Y_data[:, 1])) + 1
        self.n_ratings = Y_data.shape[0]
        self.true_list_for_each_user = true_list_for_each_user
        self.hate_list_for_each_user = hate_list_for_each_user
        self.num_recommend = num_recommend
        if Xinit is None: # new
            self.X = np.random.randn(self.n_items, K)
        else: # or from saved data
            self.X = Xinit 
        
        if Winit is None: 
            self.W = np.random.randn(K, self.n_users)
        else: # from daved data
            self.W = Winit
            
        # normalized data, update later in normalized_Y function
        self.Y_data_n = self.Y_raw_data.copy()


    def normalize_Y(self):
        '''
        normalize data by ratings of user or item
        '''
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users

        # if we want to normalize based on item, just switch first two columns of data
        else: 
            user_col = 1
            item_col = 0 
            n_objects = self.n_items

        users = self.Y_raw_data[:, user_col] 
        self.mu = np.zeros((n_objects,))
        for n in range(n_objects):
            ids = np.where(users == n)[0]
            item_ids = self.Y_data_n[ids, item_col] 
            ratings = self.Y_data_n[ids, 2]
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            # normalize
            self.Y_data_n[ids, 2] = ratings - self.mu[n]
        
    def loss(self):
        '''
        calculate the value of loss function for all ratings(contains regulization: frobenius norm)
        '''
        L = 0 
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), self.Y_data_n[i, 2]
            L += 0.5 * (rate - self.X[m, :].dot(self.W[:, n])) ** 2
        
        # take average
        L /= self.n_ratings
        # regularization, don't ever forget this 
        L += 0.5 * self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        return L 
    
    def get_items_rated_by_user(self, user_id):
        """
        get all items which are rated by user user_id, and the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:,0] == user_id)[0] 
        item_ids = self.Y_data_n[ids, 1].astype(np.int32) # indices need to be integers
        ratings = self.Y_data_n[ids, 2]
        return (item_ids, ratings)
        
        
    def get_users_who_rate_item(self, item_id):
        """
        get all users who rated item item_id and get the corresponding ratings
        """
        ids = np.where(self.Y_data_n[:,1] == item_id)[0] 
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (user_ids, ratings)
    def updateX(self):
        '''
        update the X matrix(of item) using gradient 
        '''
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            # gradient
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + self.lam * self.X[m, :]
            self.X[m, :] -= self.learning_rate * grad_xm.reshape((self.K,))
    
    def updateW(self):
        '''
        update the W matrix(of user) using gradient 
        '''
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            # gradient
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + self.lam * self.W[:, n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))

    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print('iter =', it + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)

    def pred(self, u, i):
        """ 
        predict the rating of user u for item i 
        """
        u = int(u)
        i = int(i)
        if self.user_based:
            bias = self.mu[u]
        else: 
            bias = self.mu[i]
        pred = self.X[i, :].dot(self.W[:, u]) + bias 
        # truncate if results are out of range [0, 5]
        if pred < 0:
            return 0 
        if pred > 5: 
            return 5 
        return pred 
        
    def evaluate_RMSE(self, rate_test):
        '''
        Calculate RMSE on a set
        '''

        n_tests = rate_test.shape[0]
        SE = 0
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE
    
    def recommend_list_of_each_user(self):
        '''
        Return top k recommend list for all user 
        '''

        predict_list = []
        for i in range(self.n_users):
            for j in self.true_list_for_each_user[i]:

                # ids = np.where(self.Y_test[:, 0] == i)[0]
                # items_rated_by_i = self.Y_test[ids, 1].tolist()
                recommended_items = []
                rate_of_user_i_for_item_j = self.pred(i, j)
                recommended_items.append((rate_of_user_i_for_item_j, j))
                
                # choose k negative sample
                if(len(self.hate_list_for_each_user[i]) > self.num_recommend):
                    ids = sample(range(len(self.hate_list_for_each_user[i])), self.num_recommend) 
                else:
                    ids = range(len(self.hate_list_for_each_user[i]))
                # print(ids)
                for k in range(len(ids)):
                    tmp_item = self.hate_list_for_each_user[i][ids[k]]
                    rate_of_user_i_for_tmp_item = self.pred(i, tmp_item)
                    recommended_items.append((rate_of_user_i_for_tmp_item, tmp_item))

                
                recommended_items.sort(key = lambda x : x[0], reverse=True)
                predict_list.append([x[1] for x in recommended_items[: self.num_recommend]])

        return predict_list
    

    def pred_for_user(self, user_id):
        """
        return list of recommend for each user on unrated item of that user
        """
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_data_n[ids, 1].tolist()              
        
        recommend_item = []
        for i in items_rated_by_u:
            rate_of_user_u_for_item_i = self.pred(user_id, i)
            recommend_item.append((rate_of_user_u_for_item_i, i))
        
        recommend_item.sort(key = lambda x : x[0], reverse=True)
        
        return [x[1] for x in recommend_item[: self.num_recommend]]

