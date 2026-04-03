import BaseAgent
class SarsaAgent(BaseAgent):
    def __init__(self,q_table,count_table):
        super().__init__(q_table, count_table)
    
    def update_q_value(self,state, action, reward, next_state, next_action):
        pass