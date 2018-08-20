#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:33:12 2018

@author: apple
"""

##AutoEncoders
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

##Creating the architecture of the neural network, a stacked autoencoder
class SAE(nn.Module):
    def __init__(self,):
        super(SAE, self).__init__() #get inherited methods from the module
# that makes sure that we get all the inherited classes and the methods of the parent class and the module.
        self.first_connect = nn.Linear(nb_movies,20)  
 #the first input is the number of features is the number of movies, 
 ## the second input is the number of elements in the first encoded vector
        self.second_connect = nn.Linear(20,10)
        self.third_connect = nn.Linear(10,20)
        self.fourth_connect = nn.Linear(20,nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.activation(self.first_connect(x))  #the input x is the input nb_movies
        x = self.activation(self.second_connect(x))
        x = self.activation(self.third_connect(x))
        x = self.fourth_connect(x)
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
# decay is used to reduce the learning rate after every few epochs in order to regulate the convergence

## train the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.   #number of users who rated at least one movie
    for id_user in range(nb_users):
    # can start by getting the inputvector features that contain all the ratings of all the movies 
    # given by this particular user inside the loop.
        input = Variable(training_set[id_user]).unsqueeze(0) #a single input vector
        target = input.clone()  #copy the input
        #the following means the observations contain at least one rating that is not zero
        if torch.sum(target.data > 0) > 0:  #target.data is all the ratings
            output = sae(input)
# this will make sure that we don't compute the gradient with respect to the target
            target.require_grad = False
# we only want to include in the computations the non-zero values, we don't want to include movies that the user didn't rate.         
            output[target == 0] = 0  #these values will not count in the computations of the error,
# so they won't have impact on the updates of the different weights right after having measured the error
            loss = criterion(output, target)
# this represents the average of the error but by only considering the movies that were rated
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()  #backward decides the direction to which the weight will be updated
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()  #the amount by which the weights will be updated   
    print ("epoch: "+str(epoch)+" loss: "+str(train_loss/s))           
 # the loss represents the average of the differences of the real ratings and the predicted ratings on the training set
 # epoch: 200 loss: tensor(0.9132)        
        
    
    
 ## Testing the SAE   
test_loss = 0
s = 0.   
for id_user in range(nb_users):
    input = Variable(training_set[id_user])
    target = Variable(test_set[id_user])
   # input = Variable(training_set[id_user]).unsqueeze(0) 
  #  target = Variable(test_set[id_user]).unsqueeze(0)      # the target is the real ratings of the test set
    if torch.sum(target.data > 0) > 0:  
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0  
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1. 
print ("test loss: "+str(test_loss/s))  #0.9548, less than one star, pretty powerful
# the average test loss over all users that gave at least one non-zero rating

        
        
        
        
 
        
















