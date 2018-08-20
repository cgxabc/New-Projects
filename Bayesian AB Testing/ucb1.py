#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:34:02 2018

@author: apple
"""

import numpy as np
import matplotlib.pyplot as plt
from comparing_epsilons import run_experiment as run_experiment_eps

class Bandit:
    def __init__(self, m):
        self.m = m  # the true mean
        self.mean = 10
        self.N = 0
        
    def pull(self):
        return np.random.randn() + self.m  
  
    def update(self, x):
        self.N += 1
        self.mean = (1- 1.0/self.N)*self.mean + 1.0/self.N*x
        
def ucb(mean, n, nj):
    return mean + np.sqrt(2*np.log(n) / (nj + 10e-3))
        

def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]
    
    data = np.empty(N)
    
    for i in xrange(N):
#optimisitic initial values
        j = np.argmax([ucb(b.mean, i+1, b.N) for b in bandits])
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
    c_1 = run_experiment(1.0, 2.0, 3.0, 100000)
    c_1_eps = run_experiment_eps(1.0, 2.0, 3.0, 0.1, 100000)
   
    
 # log scale plot
    plt.plot(c_1_eps, label = 'eps = 0.1')
    plt.plot(c_1, label = 'ucb1')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
# linear plot
    plt.plot(c_1_eps, label = 'eps = 0.1')
    plt.plot(c_1, label = 'ucb1')
    plt.legend()
    plt.show()  
    
    
    
    
    
    
    