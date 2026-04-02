class DoubleQLearningAgent():
    def __init__(self,q_table_B):
       self.q_table_B = q_table_B 
    
    def update_q_value(self):
        pass

    def choose_action(self,state,epsilon):
        '''
            As it uses both Q-tables to select the action with the highest value, and then updates the Q-table that was not used for action selection
        '''
        pass