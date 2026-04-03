from typing import override
from BaseAgent import BaseAgent #importing the class only not the entire file
import random

class DoubleQLearningAgent(BaseAgent):
    def __init__(self,q_table,count_table,q_table_B):
        super().__init__(q_table, count_table)
        self.q_table_B = q_table_B 
    
    def update_q_value(self):
        pass

    @override
    def choose_action(self,state,epsilon):
        '''
            Overriding the epslon-greedy policy the other methods are using as it will use the average of q-values coming from both Q-tables to select the action with the highest value
        '''
        average_q_values = self.average_q_value(state)
        if random.random() < epsilon:
            return random.choice(list(average_q_values.keys()))
        else:
            max_value = max(average_q_values.values())
            max_actions = [action for (action, value) in average_q_values.items() if value == max_value]
            return random.choice(max_actions)
        
    def average_q_value(self, state):
        return {"hit": (self.q_table[state]["hit"] + self.q_table_B[state]["hit"]) / 2, 
                "stick": (self.q_table[state]["stick"] + self.q_table_B[state]["stick"]) / 2}