from BaseAgent import BaseAgent #importing the class only not the entire file
class SarsaAgent(BaseAgent):
    def __init__(self,q_table,count_table):
        super().__init__(q_table, count_table)
    
    def update_q_value(self,state, action, reward, next_state, next_action):
        alpha = self.get_alpha(state, action)
        if next_state is None and next_action is None:
            self.q_table[state][action] += alpha * (reward - self.q_table[state][action])
        else:
            self.q_table[state][action] += alpha * (reward + self.q_table[next_state][next_action] - self.q_table[state][action])

    def training_loop(self, environment_instance, epsilon):
        #done has to be passed from env so it indicated when episode is over
        state = environment_instance.get_state()
        action = self.get_action(state, epsilon)
        done = False
        while not done:
            #step is there to take the action and return the reward and whether the episode is done
            next_state, reward, done = environment_instance.step(action) 

            if done: #If the episode is done, then we update the Q-value with the reward and break out of the loop
                next_state, next_action = None, None
            else:
                next_action = self.get_action(next_state, epsilon)

            self.increment_count(state, action) 
            self.update_q_value(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action
