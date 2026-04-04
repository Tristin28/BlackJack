from agent.BaseAgent import BaseAgent
from agent.MonteCarloAgent import MonteCarloAgent
from agent.SarsaAgent import SarsaAgent
from agent.QLearningAgent import QLearningAgent
from agent.DoubleQLearningAgent import DoubleQLearningAgent
from environment import Environment
import math 

#Helper function to initialise Q-table
def initialise_q_table():
        '''
        Since there are only 200 states and 2 actions it is more efficient to initialise the Q-table with all states and actions at the start of the program, 
        Rather than checking if a state-action pair is in the Q-table every time we want to update a Q-value or select an action.
        '''
        q_table = {}
        for player_sum in range(12, 21):
            for dealer_card in range(2, 12):
                for usable_ace in [False, True]:
                    state = (player_sum, dealer_card, usable_ace)
                    q_table[state] = {"HIT": 0.0, "STAND": 0.0}
        return q_table

def inisitalise_count_table(self):
        count_table = {}
        for player_sum in range(12, 21):
            for dealer_card in range(2, 12):
                for usable_ace in [False, True]:
                    state = (player_sum, dealer_card, usable_ace)
                    count_table[state] = {"HIT": 0, "STAND": 0}
        return count_table


def run_episodes(agent, num_episodes, exploring_starts=False):
    rewards = []

    if isinstance(agent, DoubleQLearningAgent):
        agent.q_table = initialise_q_table()
        agent.q_table_B = initialise_q_table()
        agent.count_table = inisitalise_count_table()

    else:
        agent.q_table = initialise_q_table()
        agent.count_table = inisitalise_count_table()

    for episode in range(1,num_episodes+1):
        env = Environment() 
        

if __name__ == "__main__":
    num_episodes = 100000
    q_learning_agent = QLearningAgent({}, {})
    double_q_learning_agent = DoubleQLearningAgent({}, {}, {})
    monte_carlo_agent = MonteCarloAgent({}, {})
    sarsa_agent = SarsaAgent({}, {})
    
    run_episodes(q_learning_agent, num_episodes)
    run_episodes(double_q_learning_agent, num_episodes)
    run_episodes(monte_carlo_agent, num_episodes)
    run_episodes(sarsa_agent, num_episodes)