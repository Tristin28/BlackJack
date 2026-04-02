from abc import ABC, abstractmethod
import random
class BaseAgent(ABC):
    #Note that state will be a tuple consisting of (player_sum, dealer_card, and usable_ace)
    def __init__(self,q_table,count_table,alpha,gamma):
        self.q_table = q_table #Nested-dictionary to store the Q-values for each state-action pair
        self.count_table = count_table #Nested-dictionary to store the count of how many times each state-action pair has been visited
        self.alpha = alpha #Step size, so that each update to the Q-values is a fraction of the difference between the current Q-value and the target Q-value
        self.gamma = gamma #Discount factor, so that future rewards are discounted when updating the Q-values
        

    def increment_count(self,state,action):
        #Since both tables are initialised with zeros as it is needed for TD methods, and doesnt effect MC, then i dont need any if conditions to check
        self.count_table[state][action] += 1
        
    def choose_action(self,state,epsilon):
        '''
            Note that this is not done here so that forced rules stay in environment/training control logic because it doesnt necessarily have to do with the policy
            player_sum, _, _ = state
            if player_sum < 12:
                return 'hit'
            if player_sum == 21:
                return 'stick'

            This function will represent the epsilon-greedy policy, i.e. it represents the policy improvement stage of the policy iteration algorithm
            it is shared among all other agents, because the same policy is required for all methods
        '''
        if random.random() < epsilon:
            return random.choice(list(self.q_table[state].keys()))
        else:
            #Note even though time complexity is O(n), it is still efficient because it only iterates through 2 actions
            max_value = max(self.q_table[state].values())
            max_actions = [action for (action, value) in self.q_table[state].items() if value == max_value]
            return random.choice(max_actions)
   
    @abstractmethod
    def update_q_value(self,state,action,reward,next_state):
        #This method is abstract because each method updates the Q-values differently.
        pass