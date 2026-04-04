import random
from agent.BaseAgent import BaseAgent 

class MonteCarloAgent(BaseAgent):
    def __init__(self, q_table, count_table):
         super().__init__(q_table, count_table)
        
    def update_q_value(self, state, action, reward):
        alpha = self.get_alpha(state, action)
        self.q_table[state][action] += alpha * (reward - self.q_table[state][action]) # Gamma ommited as it is 1 in this case

    def get_action(self, state, epsilon, exploring_starts, first_decision):
        player_sum, _, _ = state
        if player_sum < 12:
            return 'HIT', first_decision
        elif player_sum == 21:
            return 'STAND', first_decision
        
        if exploring_starts and first_decision:
                 return random.choice(list(self.q_table[state].keys())), False
        
        return super().choose_action(state, epsilon), False

    def run_episode(self, environment_instance, epsilon, exploring_starts=False):
        state = environment_instance.get_state()
        done = False
        first_decision = True
        episode_trace = [] # To store the state-action pairs for the current episode, which we will use to update the Q-values after the episode ends - this approach aligns with the Monte Carlo Method

        while not done:
            action, first_decision = self.get_action(state, epsilon, exploring_starts, first_decision)

            if 12 <= state[0] <= 20:
                episode_trace.append((state, action))

            next_state, reward, done = environment_instance.step(action)
            state = next_state

        return episode_trace, reward

        """
        This is how the training loop would look like in the main function:

        for k in range(1, num_episodes + 1):
        env = Environment()
        epsilon = ... # configure epsilon decay as needed
        episode_trace, reward = mc_agent.run_episode(env, epsilon, exploring_starts)

        # Ever-visit approach:
        for state, action in episode_trace:
            mc_agent.increment_count(state, action)
            mc_agent.update_q_value(state, action, reward)
        """

