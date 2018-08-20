#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:31:40 2018

@author: apple
"""

##Boltzmann Machines
##importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

## importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, 
                     engine = 'python',encoding = 'latin-1' )
#print movies.head()
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, 
                     engine = 'python',encoding = 'latin-1' )

ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, 
                     engine = 'python',encoding = 'latin-1' )
##for ratings, the first column is user id, the second column is movie id.

##prepare for the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
##the first column corresponds to users, the second corresponds to movies,
##the third column corresponds to the ratings, the fourth corresponds to timestamp
training_set = np.array(training_set, dtype = 'int')  
##get the same values, but this time into an array
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')  

##getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))  #the first column corresponds to users
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1]))) #the second column corresponds to moview

##converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
##indices in python start at 0 and movie ids start at 1
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

##converting the data into torch tensors
##two separate multi-dimensional matrics based on pytorch
training_set = torch.FloatTensor(training_set) #a torch tensor
test_set = torch.FloatTensor(test_set) #a torch tensor

##converting the ratings into binary ratings 1(liked) or 0(not liked)
##change rating = 0 to rating = -1 implying there's no rating for a movie for a specific user
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1     
        
#creating the architecture of the neural network
class RBM():
    ##initialize the parameters of our future objects
    ##all the weights will be initialized in a torched tensor, all these weights are all
    ##the parameters of the visible nodes given the hidden nodes. 
    ##The weights have to be initialized randomly according to a normal distribution
    def __init__(self, nv, nh):  #number of visible nodes and number of hidden nodes
        ## this initializes all the weights for the probabilities of the visible nodes according to the hidden nodes.
        self.W = torch.randn(nh, nv)  #initialize a tensor of nh and nv according to a normal distribution
        ## this initializes the bias for the probability of the hidden nodes given the visible nodes
        ##the first dimension corresponds to the batch and the second dimension corresponds to the bias
        self.a = torch.randn(1,nh)  #there's one bias for each hidden node; there are nh hidden nodes.
        self.b = torch.randn(1,nv)
##sample the activation of the hidden nodes
    def sample_h(self, x):
    ## make the product of two tensors.
        wx = torch.mm(x, self.W.t())  #x--the visible neuron, w--the tensor of weights
    ##of the object that will be initialized 
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
    # will return all the probabilities of the hidden neurons given the values of the visible nodes
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W) 
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
    # will return all the probabilities of the hidden neurons given the values of the visible nodes
        return p_v_given_h, torch.bernoulli(p_v_given_h)
        #the number of visible nodes is the same as the number of the movies
#v0: the input vector containing the ratings of all the movies by one user 
#vk: the visible nodes obtained after k iterations in contrastive divergence 
#ph0: thats the vector of probabilities that at first iteration the hidden nodes equal 1
# given the value of v0
    def train(self,v0, vk,ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0-phk), 0)
        
#print training_set
nv = len(training_set[0])  #number of elements in the first line, i.e. number of features for nv, 
##equal to number of movies = 1682
nh = 100 #number of hidden nodes corresponds to the number of features
batch_size = 100
rbm = RBM(nv, nh)

## training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.  #a counter, will increase after each epoch
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user: id_user + batch_size]           # the input batch of observations. 
 #i.e. the input batch of all the ratings of the users in the batch, the ratings that already existed
        v0 = training_set[id_user: id_user + batch_size]                               
# are our ratings of the movies that were already rated by the 100 users in the batch,
# don't want to touch, because need to compare
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
   ## we don't want to learn where there's no rating. only want it to train on the ratings that happened.
   ## won't be possible to update them during gibbs sampling.
            vk[v0 < 0] = v0[v0 < 0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk) #update the weights
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
# train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE 
        s += 1.
    print ('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
  ##0.25 means get a correct predictive rating three times out of four.    
 
#print torch.abs(v0[v0>=0] - vk[v0>=0])      
#print torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))   ## 0.2481       
#tensor1 = torch.abs(v0[v0>=0] - vk[v0>=0]) 
#print len(tensor1[tensor1>0]) 
#print 2623.0/10571  #0.248131681014


## Testing the RBM
test_loss = 0
s = 0.  #a counter, will increase after each epoch
for id_user in range(nb_users):
#we need this as input to get the predicted ratings of the test because we are getting these
#predicted ratings from the inputs of the training set that are used to activate the neurons of RBM
    v = training_set[id_user:id_user+1]     
#we're going to make predictions for each of the users one by one.
    vt = test_set[id_user:id_user+1]    
#the target, contains the original ratings of the test set,that is what we will compare to our predictions in the end.
# just need one step of the blind walk                    
    if len(vt[vt>=0]) > 0:  # if the number of the visible nodes containing such ratings must be larger than zero
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(v[vt>=0] - vt[vt>=0]))
# test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE 
        s += 1.
print ('test loss: '+str(test_loss/s))  #0.2394

             


















