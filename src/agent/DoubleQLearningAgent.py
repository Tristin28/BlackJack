from BaseAgent import BaseAgent #importing the class only not the entire file
class DoubleQLearningAgent(BaseAgent):
    def __init__(self,q_table,count_table,q_table_B):
        super().__init__(q_table, count_table)
        self.q_table_B = q_table_B 
    
    def update_q_value(self):
        pass

    def choose_action(self,state,epsilon):
        '''
            As it uses both Q-tables to select the action with the highest value, and then updates the Q-table that was not used for action selection
        '''
        pass