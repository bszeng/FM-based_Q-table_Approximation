
### A total of 6 parts have been added to the original code (https://github.com/farismismar/Deep-Reinforcement-Learning-for-5G-Networks/blob/master/voice/QLearningAgent.py)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bszeng
"""

import random
import numpy as np

### Added 1/6: use turicreate(https://github.com/apple/turicreate) for factorization machine 
import turicreate as tc

# Action size and bin sizes both need to be reflected to the environment settings.

class QLearningAgent:
    def __init__(self, seed,
                 learning_rate=0.2,
                 discount_factor=0.995,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.9995):
        
        self.learning_rate = learning_rate          # alpha
        self.discount_factor = discount_factor      # gamma
        self.exploration_rate = exploration_rate / exploration_decay_rate   # epsilon
        self.exploration_rate_min = 0.05
        self.exploration_decay_rate = exploration_decay_rate # d
        self.state = None
        self.action = None       
        
        self._state_size = 6 # discretized from observation
        self._action_size = 16 # check the actions in the environment        
       
        self.oversampling = 1 # for discretization.  Increasing may cause memory exhaust.
        
        # Add a few lines to caputre the seed for reproducibility.
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Discretize the continuous state space for each of the features.
        num_discretization_bins = self._state_size * self.oversampling
        
        # check the site distance configuration in the environment
        self._state_bins = [
            # User X - serv
            self._discretize_range(-350, 350, num_discretization_bins),
            # User Y - serv
            self._discretize_range(-350, 350, num_discretization_bins),
            # User X - interf
            self._discretize_range(175, 875, num_discretization_bins),
            # User Y - interf
            self._discretize_range(-350, 350, num_discretization_bins),
            # Serving BS power.
            self._discretize_range(0, 40, num_discretization_bins),
            # Interfering BS power.
            self._discretize_range(0, 40, num_discretization_bins),
        ]
        
        # Create a clean Q-Table.
        self._max_bins = max(len(bin) for bin in self._state_bins)
        num_states = (self._max_bins + 1) ** len(self._state_bins)
        self.q = np.zeros(shape=(num_states, self._action_size))
        
        ### Added 2/6 : record the explored state-action pairs
        self.explored = np.zeros(shape=(num_states, self._action_size))
        
        ### Added 3/6 : record the number of the states
        self.rows  = num_states
        
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min    
       
        self.state = self._build_state(observation) 
        
        return np.argmax(self.q[self.state, :]) # returns the action with largest Q
    
    ### Added 4/6: Q-table approximation based on factorization machine
    def factorization_machine(self):
        state_id = []
        action_id = []
        state = []
        action = []
        q_value = []
        
        # Set the Q-values of the unexplored state-action pairs to 0
        self.q = self.q * self.explored
        
        # Construct the explored experiences
        for i in range(0, self.rows):
            for j in range(0, self._action_size):
                state.append(i)
                action.append(j)
                if self.q[i][j] != 0: 
                    state_id.append(i)
                    action_id.append(j) 
                    q_value.append(self.q[i][j]) 
        
        sf = tc.SFrame({'user_id': state_id,  'item_id': action_id, 'rating': q_value})

		# Model factorization machine
        m1 = tc.factorization_recommender.create(sf, target='rating', max_iterations=500, verbose=False)
        
        # Predict all the Q-values of the Q-table
        sf_all = tc.SFrame({'user_id': state, 'item_id': action})

        self.q = ((m1.predict(sf_all)).to_numpy()).reshape((self.rows,  self._action_size))

    ### Added 5/6: Boltzmann strategy
    def boltzmann(self, observation, reward):
        next_state = self._build_state(observation)        
        state_q = self.q[next_state]
        state_q_array = np.array(state_q)
        
        # Subtract the maximum value to prevent overflow
        state_q = state_q_array.tolist()
        state_q_array = state_q_array - np.max(state_q_array)
        state_q = state_q_array.tolist()
        
        # Calculate the selection probability for each action
        act_prob  = np.exp(state_q)/np.sum(np.exp(state_q))
        
        # Select the next action based on the calculated probability
        next_action = np.random.choice(range(self._action_size),  p = act_prob)
        
        # Learn: update Q-Table based on current reward and future action.
        self.q[self.state, self.action] += self.learning_rate * \
            (reward + self.discount_factor * max(self.q[next_state, :]) - self.q[self.state, self.action])  
        
        # Record the latest explored state-action pair
        self.explored[self.state, self.action] = 1 
        
        self.state = next_state
        self.action = next_action
        
        return next_action
    
    def act(self, observation, reward):
        next_state =  self._build_state(observation)
        #state = self.state
        
        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = np.random.uniform(0, 1) <= self.exploration_rate
        if enable_exploration:
            next_action = np.random.randint(0, self._action_size)
        else:
            next_action = np.argmax(self.q[next_state])        
        
        # Learn: update Q-Table based on current reward and future action.
        self.q[self.state, self.action] += self.learning_rate * \
            (reward + self.discount_factor * max(self.q[next_state, :]) - self.q[self.state, self.action])    
        
        ### Added 6/6 : record the latest explored state-action pair
        self.explored[self.state][self.action] = 1
        
        self.state = next_state
        self.action = next_action
        return next_action

    def get_performance(self):
        return self.q.mean()

    # Private members:
    def _build_state(self, observation):
        # Discretize the observation features and reduce them to a single integer.
        state = sum(
            self._discretize_value(feature, self._state_bins[i]) * ((self._max_bins + 1) ** i)
            for i, feature in enumerate(observation)
        )
        return state
    
    def _discretize_value(self, value, bins):
        return np.digitize(x=value, bins=bins)
    
    def _discretize_range(self, lower_bound, upper_bound, num_bins):
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]
