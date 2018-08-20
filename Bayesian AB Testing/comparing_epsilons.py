#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 11:47:40 2018

@author: apple
"""

import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m):
        self.m = m  # the true mean
        self.mean = 0
        self.N = 0
        
    def pull(self):
        return np.random.randn() + self.m  
    # the first part is a sample from the standard normal distribution
    
    def update(self, x):
        self.N += 1
        self.mean = (1- 1.0/self.N)*self.mean + 1.0/self.N*x
        

def run_experiment(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    
    data = np.empty(N)
    
    for i in xrange(N):
        p = np.random.random()  # a random number p between 0 and 1
        if p < eps:
            j = np.random.choice(3)  # choose a bandit at random
        else:
            j = np.argmax([b.mean for b in bandits])  # choose the bandit with the best current sample mean
        x = bandits[j].pull()  # choose a bandit
        bandits[j].update(x)  #update the bandit with the reward we just got from x
    # for the plot
        data[i] = x    
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
# we plot the cumulative average along with bars showing each of the means 
# so we can see where our cumulative average relative to those
    #plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()
    
    for b in bandits:
        print b.mean      
    return cumulative_average

if __name__ == '__main__':
    c_1 = run_experiment(1.0, 2.0, 3.0, 0.1, 100000)
    c_05 = run_experiment(1.0, 2.0, 3.0, 0.05, 100000)
    c_01 = run_experiment(1.0, 2.0, 3.0, 0.01, 100000)
    
 # log scale plot
    plt.plot(c_1, label = 'eps = 0.1')
    plt.plot(c_05, label = 'eps = 0.05')
    plt.plot(c_01, label = 'eps = 0.01')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
# linear plot
    plt.plot(c_1, label = 'eps = 0.1')
    plt.plot(c_05, label = 'eps = 0.05')
    plt.plot(c_01, label = 'eps = 0.01')
    plt.legend()
    plt.show()     
        
        
#N = 3     
#data = [2.,3.,6.]  
#print np.cumsum(data)   #[2 5 11]
#print (np.arange(3) + 1)  #[1 2 3]
#print np.cumsum(data)/(np.arange(3) + 1)  #[2. 2.5 3.667]
        
        
        
        
        
        
        
        
        
        
        
        
        
        