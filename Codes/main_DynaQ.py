### A total of 6 parts have been added to the original code (https://github.com/farismismar/Deep-Reinforcement-Learning-for-5G-Networks/blob/master/voice/main.py)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: bszeng
"""

import os
import random
import numpy as np
import pandas as pd
from colorama import Fore, Back, Style

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import time
import datetime

from environment import radio_environment
# from QLearningAgent import QLearningAgent as QL
# from QLearningAgent import DoubleQLearningAgent as DQL
# from QLearningAgent import QlambdaLearningAgent as QlL
from QLearningAgent import DynaQLearningAgent as DynaQ

def run_agent_DynaQ(env, agent_tpye, exploration_approach, FM,  FM_freq, seed,plotting=False):
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = radio_frame
    successful = False 
    episode_successful = [] # a list to save the good episodes
    Q_values = []
    FM_done = False

    max_episode = -1
    max_reward = -np.inf
    FM_start= 0

    print('Ep.         | TS | Recv. SINR (srv) | Recv. SINR (int) | Serv. Tx Pwr | Int. Tx Pwr | Reward ')
    print('--'*54)
    
    if FM:
        FM_on = 'on'
    else:
        FM_on = 'off'
    f_n = '{}_{}_{}_FM{}_mu{}_seed{}.txt'.format(agent_tpye, MAX_EPISODES, exploration_approach, FM_on, FM_freq,seed)
    f = open(f_n,'w')
    titles = 'exploration,epsiode,timesteps,rewards,FM_done'
    print(titles)
    f.write('\n' + titles)

    # Implement the Q-learning algorithm
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()
        (_, _, _, _, pt_serving, pt_interferer) = observation        
        action = agent.begin_episode(observation)

        # Let us know how we did.
        print('{}/{} | {:.2f} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W | {:.2f} | {} '.format(episode_index, max_episodes_to_run, 
                                                                                      agent.exploration_rate,
                                                                                      0, 
                                                                                      np.nan,
                                                                                      np.nan,
                                                                                      pt_serving, pt_interferer, 
                                                                                      0, action))        
        total_reward = 0
        timestep_count = 0
        done = False
        actions = [action]        

        sinr_progress = [] # needed for the SINR based on the episode.
        sinr_ue2_progress = [] # needed for the SINR based on the episode.
        serving_tx_power_progress = []
        interfering_tx_power_progress = []       

        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            start_time = time.time()
            timestep_count += 1
            next_observation, reward, done, abort = env.step(action)
            (_, _, _, _, pt_serving, pt_interferer) = next_observation
            # make next_state the new current state for the next frame.
            #observation = next_observation
            total_reward += reward
            ### Added 1/6: Select actions by egreedy or Boltzmann
            action = agent.act(next_observation, reward, exploration_approach)
            received_sinr = env.received_sinr_dB
            received_ue2_sinr = env.received_ue2_sinr_dB            
            # Learn control policy
            successful = (total_reward > 0) and (abort == False)            
            # Let us know how we did.
            print('{}/{} | {} | {} | {:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W | {:.2f} | {} '.format(episode_index, max_episodes_to_run, 
                                                                                          seed,
                                                                                          timestep_index, 
                                                                                          received_sinr,
                                                                                          received_ue2_sinr,
                                                                                          pt_serving, pt_interferer, 
                                                                                          total_reward, action), end='')     
    
            actions.append(action)
            sinr_progress.append(env.received_sinr_dB)
            sinr_ue2_progress.append(env.received_ue2_sinr_dB)
            serving_tx_power_progress.append(env.serving_transmit_power_dBm)
            interfering_tx_power_progress.append(env.interfering_transmit_power_dBm)
            
            if abort == True:
                print('ABORTED.')
                break
            else:
                print()
        
        states_explored = np.sum(agent.explored, axis=1)
        states_explored_count = np.sum(states_explored!=0)
        init_count = np.sum(agent.q*agent.explored==0)
        total_count = agent.q.size
        sparsity = init_count/total_count
        states_explored_ratio = states_explored_count / agent.rows
        print(str(agent_tpye) +', Exploration: ' +str(exploration_approach)+', State-action explored: ' + str(total_count-init_count) + ', States explored ratio: ' + str(states_explored_ratio) + ', Q-table sparsity: ' + str(sparsity))

        if (FM == True) and (states_explored_ratio - FM_start >= FM_freq): 
            print('factorzation_machine is activated.')
            agent.factorization_machine()
            FM_start = states_explored_ratio
            FM_done = True
        if (successful == True) and (abort == False):
            print(Fore.GREEN + 'SUCCESS.  Total reward = {}.'.format(total_reward))
            print(Style.RESET_ALL)  
            if (total_reward > max_reward):
                max_reward, max_episode = total_reward, episode_index  
        else:
            reward = 0
            print(Fore.RED + 'FAILED TO REACH TARGET.')
            print(Style.RESET_ALL)
        
        results ='%s,%d,%d,%d,%s' % (exploration_approach, episode_index, timestep_count, total_reward,FM_done)
        f.write('\n' + results)
        FM_done = False
    f.close()       

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))

########################################################################################
    
radio_frame = 20
MAX_EPISODES = 100000 
np.random.seed(0)
seeds = np.random.randint(1,100,10).tolist() 
Approximation_threshold =  {0.02} #{0.01, 0.02, 0.03, 0.04, 0.05}

for FM_threshold in Approximation_threshold:    
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed) 
        env = radio_environment(seed=seed)
        agent = DynaQ(seed=seed, noPlanning=100)
        run_agent_DynaQ(env, agent_tpye='DynaQ', exploration_approach='boltzmann', FM=True, FM_freq = FM_threshold, seed=seed)
        
        random.seed(seed)
        np.random.seed(seed) 
        env = radio_environment(seed=seed)
        agent= DynaQ(seed=seed, noPlanning=100)
        run_agent_DynaQ(env, agent_tpye='DynaQ', exploration_approach='boltzmann', FM=False, FM_freq = FM_threshold, seed=seed) 
        
        random.seed(seed)
        np.random.seed(seed)
        env = radio_environment(seed=seed)
        agent = DynaQ(seed=seed, noPlanning=100)
        run_agent_DynaQ(env, agent_tpye='DynaQ', exploration_approach='egreedy', FM=False, FM_freq = FM_threshold, seed=seed) 

        random.seed(seed)
        np.random.seed(seed)
        env = radio_environment(seed=seed)
        agent = DynaQ(seed=seed, noPlanning=100)
        run_agent_DynaQ(env, agent_tpye='DynaQ', exploration_approach='egreedy', FM=True, FM_freq = FM_threshold, seed=seed) 


########################################################################################
