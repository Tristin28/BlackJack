class DoubleQLearningAgent():
    #Note that state will be a tuple consisting of (player_sum, dealer_card, and usable_ace)
    def __init__(self,q_table,count_table,gamma):
        self.q_table = q_table #Nested-dictionary to store the Q-values for each state-action pair
        self.count_table = count_table #Nested-dictionary to store the count of how many times each state-action pair has been visited
    
    def update_q_value(self):
        pass