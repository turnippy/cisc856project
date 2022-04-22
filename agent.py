#CISC856 Winter 2022
#Project Group 6
#Tong Liu, Peash Saha

import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from itertools import product

import Environment

class Agent:
  def __init__(self, env, k=2000, epsilon=0.5, alpha=0.8, gamma=0.8):
    '''
    initialize with:
      environment
      type of update algorithm
      k = number of epochs
      epsilon
      alpha = learning rate
      gamma = discount factor
    '''
    self.env = env
    #self.alg = alg
    self.k = k
    self.epsilon = epsilon
    self._epsilon = epsilon #used to track startingg epsilon value
    self.alpha = alpha
    self.gamma = gamma
    self.initialize_q_table(self.env.n, self.env.c, self.env.t)
  
  def initialize_q_table(self, n, c, t):
    #q-table is maintained as a double hash table
    #the keys of the outer hash is the state tuple
    #the keys of the inner hash the action tuple in the given state
    self.q_table = {}
    #first, generate all possible states. by the problem descriptions, we assume
    #that S_(i,t) >= 0 for all i <= N, for all t <= T

    all_states = [i for i in product(range(-(t+1), t+2), repeat=n)]
    all_actions = []
    for item in product(range(0, c+1), repeat=n):
      if sum(item) <= c and sum(item) > 0:
        all_actions.append(item)

    for tup in all_states:
      self.q_table[tup] = {}
      for act in all_actions:
        self.q_table[tup][act] = 0

  def action_selection(self, curr):
    curr = tuple(curr)
    if random.random() < self.epsilon:
      return random.choice(list(self.q_table[curr].keys()))

    best_q = float('-inf')
    best_actions = []
    for a in self.q_table[curr].keys():
      tmp = self.q_table[curr][a]
      if tmp > best_q:
        best_q = tmp
        best_actions = [a]
      if tmp == best_q:
        best_actions.append(a)
    return random.choice(best_actions)

  def train(self, k, static_epsilon=False):
    training = []
    self.epsilon = self._epsilon
    for t in range(1, k+1):
      if not static_epsilon:
        self.calc_epsilon(t, k)
      episode = self.train_episode(t)
      training.append(episode)
    return training

  def calc_epsilon(self, current, total):
    #according to Yu et al 2021
    self.epsilon = 0.5 / (1 + np.exp(10*(current-0.4*total)/total))

  def get_params(self):
    ret = ''
    return ret
    
class Sarsa(Agent):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    
  def update_qtable(self, curr, action, reward, next_s, next_a):
    curr = tuple(curr)
    next_s = tuple(next_s)
    self.q_table[curr][action] += self.alpha * (reward + self.gamma * self.q_table[next_s][next_a] - self.q_table[curr][action])    
    
  def train_episode(self, epoch):
    curr = (0,) * self.env.n
    action = self.action_selection(curr)
    done = False
    #history = []
    total_cost = 0
    while not done:
        next_state, reward, t_step = self.env.take_action(action)
        next_action = self.action_selection(next_state)
        self.update_qtable(curr, action, reward, next_state, next_action)
        #history.append((curr, action, reward))
        #print('reward for this move: {}, next state: {}'.format(reward, next))
        total_cost += reward
        state = next
        if t_step >= self.env.t:
          self.env.reset()
          done = True
    if epoch % 50 == 0:
      print('Training Episode {} cost: {}.'.format(epoch, total_cost))
    return total_cost

class Qlearning(Agent):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    
  def update_qtable(self, curr, action, reward, next):
    curr = tuple(curr)
    next = tuple(next)
    temp = []
    for a in self.q_table[next].keys():
      temp.append(self.q_table[next][a])
    self.q_table[curr][action] += self.alpha * (reward + self.gamma * max(temp) - self.q_table[curr][action])    
    
  def train_episode(self, epoch):
    curr = (0,) * self.env.n
    done = False
    #history = []
    total_cost = 0
    while not done:
        action = self.action_selection(curr)
        next, reward, t_step = self.env.take_action(action)
        self.update_qtable(curr, action, reward, next)
        #history.append((curr, action, reward))
        #print('reward for this move: {}, next state: {}'.format(reward, next))
        total_cost += reward
        state = next
        if t_step >= self.env.t:
          self.env.reset()
          done = True
    if epoch % 50 == 0:
      print('Training Episode {} cost: {}.'.format(epoch, total_cost))
    return total_cost
