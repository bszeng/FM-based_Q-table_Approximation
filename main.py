
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
from QLearningAgent import QLearningAgent as QLearner

def run_agent_tabular(env, exploration_approach, FM,  FM_freq, seed,plotting=False):
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = radio_frame
    successful = False    
    episode_successful = [] # a list to save the good episodes
    Q_values = []
    recording = pd.DataFrame()
    FM_done = False

    max_episode = -1
    max_reward = -np.inf
    FM_start= 0

    print('Ep.         | TS | Recv. SINR (srv) | Recv. SINR (int) | Serv. Tx Pwr | Int. Tx Pwr | Reward ')
    print('--'*54)
    
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
        episode_q = []

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
            if exploration_approach == 'egreedy':
                action = agent.act(next_observation, reward)
            if exploration_approach =='boltzmann':
                action = agent.boltzmann(next_observation, reward)

            received_sinr = env.received_sinr_dB
            received_ue2_sinr = env.received_ue2_sinr_dB
            
            # Learn control policy
            q = agent.get_performance()
            episode_q.append(q)
                            
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
        
        ### Added 2/6: Caclulate the number of the explored states.
        states_explored = np.sum(agent.explored, axis=1)
        states_explored_count = np.sum(states_explored!=0)

        ### Added 3/6: Cacluate the sparsity of Q-table
        init_count = np.sum(agent.q*agent.explored==0)
        total_count = agent.q.size
        sparsity = init_count/total_count
        states_explored_ratio = states_explored_count / agent.rows
        print('Exploration: ' +str(exploration_approach)+', State-action explored: ' + str(total_count-init_count) + ', States explored ratio: ' + str(states_explored_ratio) + ', Q-table sparsity: ' + str(sparsity))

        # at the level of the episode end
        q_z = np.mean(episode_q)

        ### Added 4/6:  Q-table approximation
        if (FM == True) and (states_explored_ratio - FM_start >= FM_freq): 
            print('factorzation_machine is activated.')
            agent.factorization_machine()
            FM_start = states_explored_ratio
            FM_done = True

        ### Added 5/6: Record the results of each episode       
        end_time = time.time()
        duration = 1000. * (end_time - start_time)
        result = {'explore': exploration_approach, 'epsiode': episode_index, 'timesteps': timestep_count, 'rewards': total_reward, 'States_explored_ratio': states_explored_ratio,  'sparsity':sparsity, 'FM_done': FM_done, 'duration': duration }
        recording = recording.append(result, ignore_index=True)
        FM_done = False

        if (successful == True) and (abort == False):
            print(Fore.GREEN + 'SUCCESS.  Total reward = {}.'.format(total_reward))
            print(Style.RESET_ALL)  
            if (total_reward > max_reward):
                max_reward, max_episode = total_reward, episode_index  
        else:
            reward = 0
            print(Fore.RED + 'FAILED TO REACH TARGET.')
            print(Style.RESET_ALL)

        Q_values.append(q_z)
        
   ### Added 6/6: Save data
    filename = 'figures/recording_{}_{}_FM{}_Freq{}_seed{}.csv'.format(MAX_EPISODES, exploration_approach, FM, FM_freq,seed)
    recording.to_csv(filename, index=True)    

    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes.".format(max_episodes_to_run))
    

########################################################################################
    
radio_frame = 20
seeds = np.arange(51).tolist()

MAX_EPISODES = 100000

Approximation_threshold = {0.01, 0.02, 0.03, 0.04, 0.05}

for FM_threshold in Approximation_threshold:
    
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed) 
        env = radio_environment(seed=seed)
        agent = QLearner(seed=seed)
        run_agent_tabular(env, exploration_approach='boltzmann', FM=True, FM_freq = FM_threshold, seed=seed)
        
        random.seed(seed)
        np.random.seed(seed) 
        env = radio_environment(seed=seed)
        agent = QLearner(seed=seed)
        run_agent_tabular(env, exploration_approach='boltzmann', FM=False, FM_freq = FM_threshold, seed=seed) 
        
        random.seed(seed)
        np.random.seed(seed)
        env = radio_environment(seed=seed)
        agent = QLearner(seed=seed)
        run_agent_tabular(env, exploration_approach='egreedy', FM=False, FM_freq = FM_threshold, seed=seed) 

        random.seed(seed)
        np.random.seed(seed)
        env = radio_environment(seed=seed)
        agent = QLearner(seed=seed)
        run_agent_tabular(env, exploration_approach='egreedy', FM=True, FM_freq = FM_threshold, seed=seed) 


########################################################################################