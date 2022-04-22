#CISC856 Winter 2022
#Project Group 6
#Tong Liu, Peash Saha

import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment
from agent import Agent, Sarsa, Qlearning

def plot(data, title):
    #graphs the number of moves in the solution vs epoch number
    fig,ax = plt.subplots()
    #graphs each column of each history as its own plot
    #for key in data.keys():
        #ax.plot(histories[key])
        #ax.plot(data[key], label='{}'.format(key))
    plt.plot(data)
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Objective')
    ax.set_title(title, y=1.1)
    plt.savefig('{}.png'.format(title))
    
def plot_series(x, y, title, xlab, ylab):
    #graphs the number of moves in the solution vs epoch number
    fig,ax = plt.subplots()
    plt.plot(x,y)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
    plt.savefig('{}.png'.format(title))
    
def standardize(data):
    result = np.array(data)
    #result /= -10000
    np.round(result, 1)
    return result

if __name__ == '__main__':
    accessibility_costs = np.array([200,250,300,350,400,450,500,550,600,650])
    algorithms = ['sarsa', 'ql']
    e = Environment(n=3,c=1,t=6,gamma=0.5,ac=accessibility_costs,dca=2.04,dcb=0.24)

    # Epsilon comparison
    epsilon_s = [True, False]
    for epi in epsilon_s:
        ag1 = Sarsa(env=e, epsilon=0.5, alpha=0.8, gamma=0.2)
        start = time.time()
        res1 = standardize(ag1.train(k=2000, static_epsilon=epi))
        end = time.time()
        title1 = 'SARSA, static_explore = {}, running time = {} ms'.format(epi, round(end-start,2))
        plot(res1, title1)
        
        ag2 = Qlearning(env=e, epsilon=0.5, alpha=0.8, gamma=0.2)
        start = time.time()
        res2 = standardize(ag2.train(k=2000,static_epsilon=epi))
        end = time.time()
        title2 = 'QLearning, static_explore = {}, running time = {} ms'.format(epi, round(end-start,2))
        plot(res2, title2)
    
    # Alpha comparison
    alpha_s = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for alpha in alpha_s:
        ag1 = Sarsa(env=e, epsilon=0.5, alpha=alpha, gamma=0.2)
        start = time.time()
        res1 = standardize(ag1.train(k=2000))
        end = time.time()
        title1 = 'SARSA, alpha = {}, running time = {} ms'.format(alpha, round(end-start,2))
        plot(res1, title1)
        
        ag2 = Qlearning(env=e, epsilon=0.5, alpha=alpha, gamma=0.2)
        start = time.time()
        res2 = standardize(ag2.train(k=2000))
        end = time.time()
        title2 = 'QLearning, alpha = {}, running time = {} ms'.format(alpha, round(end-start,2))
        plot(res2, title2)
    
    # Gamma comparison
    gamma_s = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for gamma in gamma_s:
        ag1 = Sarsa(env=e, epsilon=0.5, alpha=0.8, gamma=gamma)
        start = time.time()
        res1 = standardize(ag1.train(k=2000))
        end = time.time()
        title1 = 'SARSA, gamma = {}, running time = {} ms'.format(gamma, round(end-start,2))
        plot(res1, title1)
        
        ag2 = Qlearning(env=e, epsilon=0.5, alpha=0.8, gamma=gamma)
        start = time.time()
        res2 = standardize(ag2.train(k=2000))
        end = time.time()
        title2 = 'QLearning, gamma = {}, running time = {} ms'.format(gamma, round(end-start,2))
        plot(res2, title2)
    '''
    # Impact of |N|
    
    n_s = [3,4,5,6]
    t_s = [6,8]
    
    for t_ in t_s:
        title = 'Comparison of |N| with T = {} using SARSA'.format(t_)
        xs = []
        ys = []
        for n_ in n_s:
            e = Environment(n=n_,c=1,t=t_,gamma=0.5,ac=accessibility_costs,dca=2.04,dcb=0.24)
            ag1 = Sarsa(env=e, epsilon=0.5, alpha=0.8, gamma=0.2)
            start = time.time()
            res1 = standardize(ag1.train(k=2000))
            end = time.time()
            xs.append(n_)
            ys.append(round(end-start, 2))
        plot_series(xs, ys, title, 'Number of affected areas', 'Execution time')
        
        title = 'Comparison of |N| with T = {} using QL'.format(t_)
        xs = []
        ys = []
        for n_ in n_s:
            e = Environment(n=n_,c=1,t=t_,gamma=0.5,ac=accessibility_costs,dca=2.04,dcb=0.24)
            ag1 = Qlearning(env=e, epsilon=0.5, alpha=0.8, gamma=0.2)
            start = time.time()
            res1 = standardize(ag1.train(k=2000))
            end = time.time()
            xs.append(n_)
            ys.append(round(end-start, 2))
        plot_series(xs, ys, title, 'Number of affected areas', 'Execution time')
       
    # Impact of T 
    
    n_s = [3,5]
    t_s = [6,7,8,9,10]
    for n_ in n_s:
        title = 'Comparison of T with |N| = {} using SARSA'.format(n_)
        xs = []
        ys = []
        for t_ in t_s:
            e = Environment(n=n_,c=1,t=t_,gamma=0.5,ac=accessibility_costs,dca=2.04,dcb=0.24)
            ag1 = Sarsa(env=e, epsilon=0.5, alpha=0.8, gamma=0.2)
            start = time.time()
            res1 = standardize(ag1.train(k=2000))
            end = time.time()
            xs.append(t_)
            ys.append(round(end-start, 2))
        plot_series(xs, ys, title, 'Number of planning periods', 'Execution time')
        
        title = 'Comparison of T with |N| = {} using QL'.format(n_)
        xs = []
        ys = []
        for t_ in t_s:
            e = Environment(n=n_,c=1,t=t_,gamma=0.5,ac=accessibility_costs,dca=2.04,dcb=0.24)
            ag1 = Qlearning(env=e, epsilon=0.5, alpha=0.8, gamma=0.2)
            start = time.time()
            res1 = standardize(ag1.train(k=2000))
            end = time.time()
            xs.append(t_)
            ys.append(round(end-start, 2))
        plot_series(xs, ys, title, 'Number of planning periods', 'Execution time')
        '''