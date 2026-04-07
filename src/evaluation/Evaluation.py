from agent.BaseAgent import BaseAgent
from agent.MonteCarloAgent import MonteCarloAgent
from agent.SarsaAgent import SarsaAgent
from agent.QLearningAgent import QLearningAgent
from agent.DoubleQLearningAgent import DoubleQLearningAgent, get_average_q_table
from Environment import Environment
import math 

#Helper function to initialise Q-table
def initialise_q_table():
        '''
        Since there are only 180 states and 2 actions it is more efficient to initialise the Q-table with all states and actions at the start of the program, 
        Rather than checking if a state-action pair is in the Q-table every time we want to update a Q-value or select an action.
        '''
        q_table = {}
        for player_sum in range(12, 21):
            for dealer_card in range(2, 12):
                for usable_ace in [False, True]:
                    state = (player_sum, dealer_card, usable_ace)
                    q_table[state] = {"HIT": 0.0, "STAND": 0.0}
        return q_table

def inisitalise_count_table():
        count_table = {}
        for player_sum in range(12, 21):
            for dealer_card in range(2, 12):
                for usable_ace in [False, True]:
                    state = (player_sum, dealer_card, usable_ace)
                    count_table[state] = {"HIT": 0, "STAND": 0}
        return count_table

def get_epsilon(config, episode):
    if config == "fixed_0.1":
        return 0.1
    elif config == "1_over_k":
        return 1 / episode
    elif config == "exp_1000":
        return math.exp(-episode / 1000)
    elif config == "exp_10000":
        return math.exp(-episode / 10000)

def run_episodes(agent, config, num_episodes=100000, exploring_starts=False):
    history = []
    wins, losses, draws = 0, 0, 0

    if isinstance(agent, DoubleQLearningAgent):
        agent.q_table = initialise_q_table()
        agent.q_table_B = initialise_q_table()
        agent.count_table = inisitalise_count_table()

    else:
        agent.q_table = initialise_q_table()
        agent.count_table = inisitalise_count_table()

    for episode in range(1,num_episodes+1):
        env = Environment() #Creating a new instance because the instiate game (resetting the game) is inside the constructor 
        epsilon = get_epsilon(config, episode)

        if isinstance(agent, MonteCarloAgent):
            episode_trace, reward = agent.run_episode(env, epsilon, exploring_starts)
            
            for state, action in episode_trace:
                agent.increment_count(state, action)
                agent.update_q_value(state, action, reward)
        else:
            reward = agent.run_episode(env, epsilon)
            
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1
        
        if episode % 1000 == 0:
            history.append((episode, wins, losses, draws))
            #print(f"Episode: {episode}, Wins: {wins}, Losses: {losses}, Draws: {draws}")
            wins, losses, draws = 0, 0, 0 #Resetting the counts 
        
    if isinstance(agent, DoubleQLearningAgent):
        avg_q_table = get_average_q_table(agent.q_table, agent.q_table_B)
        visited_pairs, num_visited_pairs = get_visited_pairs_and_count(agent.count_table)
        return history, avg_q_table, agent.count_table, visited_pairs, num_visited_pairs
    else:
        visited_pairs, num_visited_pairs = get_visited_pairs_and_count(agent.count_table)
        return history, agent.q_table, agent.count_table, visited_pairs, num_visited_pairs

def get_visited_pairs_and_count(count_table):
    '''
        Helper function which returns how many different state-action pairs were explored at least once and the state-action pairs themselves
    '''
    visited_pairs = []

    for state in count_table:
        for action in count_table[state]:
            if count_table[state][action] >= 1:
                visited_pairs.append((state, action))

    return visited_pairs, len(visited_pairs)