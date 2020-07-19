from random import sample 
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
from ranking import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class Naive_bayes:
    """docstring for CF"""
    def __init__(self, 
                Y_data, 
                Y_test, 
                true_list_for_each_user,
                hate_list_for_each_user,
                num_recommend = 20, 
                ):
        '''
        initialize parameter for collaborative_filtering algorithm
        n_users, n_items: number of user and item
        self.mu: array contains the average ratings point of all user, mu[i] is mean point of user i
        Ybar_data: data after convert to sparse matrix
        num_recommend: the number of item recommended for each user
        true_list_for_each_user: true list recommend of each user in test set
        hate_list_for_each_user: the list of hated items for each user in test set
        '''

        self.Y_data = Y_data 
        self.Y_test = Y_test

        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1 
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1
        self.num_recommend = num_recommend
        self.true_list_for_each_user = true_list_for_each_user
        self.hate_list_for_each_user = hate_list_for_each_user
                
    def fit(self):
        # Convert each rating of user to document form in which the document contains two word: id of user and id of items
        data = []
        for i in range(self.Y_data.shape[0]):
            data.append(str(self.Y_data[i][0]) + ' ' + str(self.Y_data[i][1]))
        vectorizer = CountVectorizer()
    
        # Create input(tf-idf vector) and output for Muitinomial Naive Bayes
        inputs = vectorizer.fit_transform(data)
        # print(len(vectorizer.get_feature_names()))
        print(vectorizer.vocabulary_.get('0'))
        outputs = self.Y_data[:, 2]
        # print(inputs)
        # print(inputs.shape)

        # print(outputs)
        classifier = MultinomialNB()
        classifier.fit(inputs, outputs)
        
        self.classifier = classifier
        self.vectorizer = vectorizer
        
    def pred(self, u, i):
        '''
        Predict the ratings of user u for item i
        '''
        data = [str(u) + ' ' + str(i)]
        data = self.vectorizer.transform(data)
        model = self.classifier
        ratings = model.predict(data)[0]
        # print(ratings)
        return ratings
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
        
   