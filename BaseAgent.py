import random
class BaseAgent():
    #Note that state will be a tuple consisting of (player_sum, dealer_card, and usable_ace)
    def __init__(self,q_table,count_table):
        self.q_table = q_table #Nested-dictionary to store the Q-values for each state-action pair
        self.count_table = count_table #Nested-dictionary to store the count of how many times each state-action pair has been visited

    def increment_count(self,state,action):
        #Since both tables are initialised with zeros as it is needed for TD methods, and doesnt effect MC, then i dont need any if conditions to check
        self.count_table[state][action] += 1
        
    def choose_action(self,state,epsilon):
        '''
            
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
   
    def get_alpha(self, state, action):
        #Since each TD method has to use that we can also use it for MC(which would be the every visit approach) so that we have the same learning rate for all methods.
        return 1/(1+self.count_table[state][action])