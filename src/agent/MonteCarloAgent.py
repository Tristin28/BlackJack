from BaseAgent import BaseAgent #importing the class only not the entire file
class MonteCarloAgent(BaseAgent):
    def __init__(self,q_table,count_table):
         super().__init__(q_table, count_table)
        
    def update_q_value(self):
        pass

    def exploring_starts(self):
        #This is another type of policy improvement method, which only MC is going to implement
        pass
    
    def training_loop(self, environment_instance, epsilon, episodes):
        for episode in range(episodes):
            state = environment_instance.get_state()
            done = False
            episode_history = [] # To store the sequence of (state, action, reward) for the episode

            while not done:
                action = self.get_action(state, epsilon)
                