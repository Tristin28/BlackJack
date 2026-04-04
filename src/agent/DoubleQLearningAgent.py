from typing import override
from BaseAgent import BaseAgent #importing class because if not like this then the module will be passed inside the current agent class
import random

class DoubleQLearningAgent(BaseAgent):
    def __init__(self,q_table,count_table,q_table_B):
        super().__init__(q_table, count_table)
        self.q_table_B = q_table_B 
    
    def update_q_value(self,state, action, reward, next_state):
        best_action = None
        alpha = self.get_alpha(state, action)
        if random.random() < 0.5:
            best_action, _ = self.get_greedy_action_and_value(next_state, self.q_table_B)
            if best_action is not None:
                self.q_table[state][action] += alpha * (reward + self.q_table[next_state][best_action] - self.q_table[state][action])
        else:
            best_action, _ = self.get_greedy_action_and_value(next_state, self.q_table)
            if best_action is not None:
                self.q_table_B[state][action] += alpha * (reward + self.q_table_B[next_state][best_action] - self.q_table_B[state][action])
        
    @override
    def choose_action(self,state,epsilon):
        '''
            Overriding the epslon-greedy policy the other methods are using as it will use the average of q-values coming from both Q-tables to 
            select the action with the highest value
        '''
        average_q_values = self.average_q_value(state)
        if random.random() < epsilon:
            return random.choice(list(average_q_values.keys()))
        else:
            max_value = max(average_q_values.values())
            max_actions = [action for (action, value) in average_q_values.items() if value == max_value]
            return random.choice(max_actions)
        
    def average_q_value(self, state):
        return {"HIT": (self.q_table[state]["HIT"] + self.q_table_B[state]["HIT"]) / 2, 
                "STAND": (self.q_table[state]["STAND"] + self.q_table_B[state]["STAND"]) / 2}
    
<<<<<<< HEAD
    def training_loop(self, environment_instance, epsilon):
=======

    def run_episode(self, environment_instance, epsilon):
>>>>>>> 7be3ca6 (changes)
        state = environment_instance.get_state()
        done = False
        while not done:
            action = self.get_action(state, epsilon)
            next_state, reward, done = environment_instance.step(action)
            self.update_q_value(state, action, reward, next_state)
            state = next_state
