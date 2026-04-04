import random
class BaseAgent():
    #Note that state will be a tuple consisting of (player_sum, dealer_card, and usable_ace)
    def __init__(self,q_table,count_table):
        self.q_table = q_table #Nested-dictionary to store the Q-values for each state-action pair
        self.count_table = count_table #Nested-dictionary to store the count of how many times each state-action pair has been visited

    def increment_count(self,state,action):
        #Since both tables are initialised with zeros as it is needed for TD methods, and doesnt effect MC, then i dont need any if conditions to check
        if state is not None and action is not None:
            self.count_table[state][action] += 1
    
    def get_greedy_action_and_value(self, state, table):
        '''
            Helper function which will be used for both the epsilon-greedy policy and the greedy policy for Q-Learning and Double Q-Learning,
            as it will return the action with the highest Q-value for a given state, and the value of that action.
        '''
        #Note even though time complexity is O(n) for max(), it is still efficient because it only iterates through 2 actions
        max_value = max(table[state].values())
        max_actions = [action for action, value in table[state].items() if value == max_value]
        best_action = random.choice(max_actions)
        return best_action, max_value

    def choose_action(self,state,epsilon):
        '''
            This function will represent the epsilon-greedy policy, i.e. it represents the policy improvement stage of the policy iteration algorithm
            it is shared among all other agents, because the same policy is required for all methods
        '''
        if random.random() < epsilon:
            return random.choice(list(self.q_table[state].keys()))
        else:
            return self.get_greedy_action_and_value(state, self.q_table)[0]
    
    # I added an exploring_starts argument which would help for the MonteCarlo Agent
    def get_action(self, state, epsilon):
        '''
            This function is only created to seperate the logic from the hard coded rules from the epsilon-greedy policy.
        '''
        player_sum, _, _ = state
        if player_sum < 12:
            return 'hit'
        elif player_sum == 21:
            return 'stand'
        else:
            return self.choose_action(state, epsilon)
   
    def get_alpha(self, state, action):
        #Since each TD method has to use that we can also use it for MC(which would be the every visit approach) so that we have the same learning rate for all methods.
        return 1/(1+self.count_table[state][action])
    
    def initialise_q_table(self, state):
        '''
        Since there are only 200 states and 2 actions it is more efficient to initialise the Q-table with all states and actions at the start of the program, 
        Rather than checking if a state-action pair is in the Q-table every time we want to update a Q-value or select an action.
        '''
        for player_sum in range(12, 21):
            for dealer_card in range(2, 12):
                for usable_ace in [True, False]:

                    state = (player_sum, dealer_card, usable_ace)

                    self.q_table[state] = {"HIT": 0.0, "STAND": 0.0}
                    self.count_table[state] = {"HIT": 0, "STAND": 0}