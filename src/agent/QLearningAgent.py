<<<<<<< HEAD
from agent.BaseAgent import BaseAgent 
=======
import BaseAgent

>>>>>>> e2d6336 (Added QLearning)
class QLearningAgent(BaseAgent):
    def __init__(self,q_table,count_table):
        super().__init__(q_table, count_table)
        
    def update_q_value(self,state, action, reward, next_state):
        alpha = self.get_alpha(state, action)
        _, best_next_value = self.get_greedy_action_and_value(next_state, self.q_table)
        self.q_table[state][action] += alpha * (reward + best_next_value - self.q_table[state][action]) # Gamma ommited as it is 1 in this case

    def run_episode(self, environment_instance, epsilon):
        state = environment_instance.get_state()
        done = False
        while not done:
            action = self.get_action(state, epsilon)
            next_state, reward, done = environment_instance.step(action)

            if 12 <= state[0] <= 20:
                self.update_q_value(state, action, reward, next_state)
            
            state = next_state
        
