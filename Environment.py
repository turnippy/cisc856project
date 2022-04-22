#CISC856 Winter 2022
#Project Group 6
#Tong Liu, Peash Saha

import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from itertools import product

class Environment:
  def __init__(self, n, c, t, gamma, ac, dca, dcb, demand=1):
    '''
    initialize with:
      n = number of affected areas
      c = capacities of response centers
      t = number of planning periods
      gamma
      ac = accessibility cost parameter
      dca = deprivation cost parameter 'a'
      dcb = deprivation cost parameter 'b'
      demand = 1 for all affected areas
      initial state = [0] * n (zero resources allocated intiially)
      initial time step = 1
    '''
    self.n = n
    self.c = c
    self.t = t #time horizon
    self.L = 72 / t #length of each planning period
    self.gamma = gamma
    self.ac = ac #vector with AC cost param of each affected area
    self.dca = dca
    self.dcb = dcb
    self.demand = np.array([demand] * n) #default=1 for all affected areas
    self.reset()

  def reset(self):
    self.state = np.zeros(self.n, dtype=int)
    self.curr_time = 1
    
  def take_action(self, action):
    '''
    action is a vector (a_1, a_2, ..., a_n),
    where each a_i represents the resource allocation to the 
    i-th affected area, given a total of 'n' affected areas

    demand is a vector (d_1, d_2, ..., d_n),
    where each d_i represents the demand for resources of the
    i-th affected area, given a total of 'n' affected areas

    Following figure (1):
    S_(i,t+1) = S_(i,t) - Y_(i,t) + D_(i,t)
    where Y_(i,t) denotes the resource allocation at time t for the i-th area,
    where D_(i,t) denotes the demand at time t for the i-th area.

    Advances state based on action and increments time step.
    
    Returns current state, reward, and time step.
    '''
    self.check_action(action)
    reward = self.calc_cost(action, self.state, self.get_next(self.state, action), self.curr_time)
    self.state = np.add(self.state, np.array(action) * -1)
    self.state = np.add(self.state, self.demand)
    self.curr_time += 1

    return copy.deepcopy(self.state), reward, self.curr_time

  def get_next(self, curr, action):
    next = copy.deepcopy(np.add(curr, np.array(action) * -1))
    next = np.add(next, self.demand)
    return next

  def check_action(self, action):
    '''
    checks if action and demand are legal under the given environment parameters
    '''
    if len(action) != self.n:
      raise ValueError('length of action tuple must equal number of affected areas')
    if sum(action) > self.c:
      raise ValueError('resource allocation must not exceed capacity')

  def calc_AC(self, action):
    #Omega function
    cost = 0
    for i in range(len(action)):
      cost += action[i] * self.ac[i]
    #print('\tAC: {}'.format(cost))
    return cost

  def calc_SDCC(self, state):
    #Tau function
    cost = 0
    for s in state:
      #approximation according to Yu et al. 2018:
      #cost += self.gamma * (self.L * (s + 1)) - self.gamma * (self.L * (s))
      #calculation according to equation 4 in Yu et al. 2018
      cost += (np.exp(self.L * self.dcb)-1) * np.exp(self.dca + (self.dcb*self.L*s))
    #print('\tstate: {}, sdcc: {}'.format(state,cost))
    return cost

  def calc_cost(self, action, state, next, time):
    '''
    calculates the reward for the chosen action in the given state and time step
    action must be a vector with size = n
    state must be a vector with size = n
    next must be a vector with size = n
    time must be an integer value, where 0 < time <= horizon+1
    '''
    next = self.get_next(state, action)
    reward = 0
    if time == 1:
      #following equation (7)
      reward -= self.calc_AC(action)
      reward -= self.calc_SDCC(state)
      reward -= self.calc_SDCC(next)
    elif time == (self.t + 1):
      #following equation (9)
      reward -= self.calc_AC(action)
      reward -= self.calc_SDCC(next)
    else:
      #following equation (8)
      reward -= self.calc_AC(action)
      reward -= self.calc_SDCC(next)
    return reward
