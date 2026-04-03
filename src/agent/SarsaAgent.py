import BaseAgent
class SarsaAgent(BaseAgent):
    def __init__(self,q_table,count_table):
        super().__init__(q_table, count_table)
    
    def update_q_value(self,state, action, reward, next_state, next_action):
        self.q_table[state][action] += self.get_alpha() * (reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action])

    def training_loop(self, environment_instance,epsilon):
        #Need some sort of function which gives me a begining state from environment instance i.e. state = environment_instance.get_initial_state() or something like that
        pass