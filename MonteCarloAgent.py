import BaseAgent
class MonteCarloAgent(BaseAgent):
    def __init__(self,q_table,count_table):
         super().__init__(q_table, count_table)
        
    def update_q_value(self):
        pass

    def exploring_starts(self):
        #This is another type of policy improvement method, which only MC is going to implement
        pass