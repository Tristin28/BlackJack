import random
#Should we use same object but have a method which will reset it to keep same object instead of having to create a new one for each episode?
class Environment:
    def __init__(self):
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []
        self.reward = None
        self.__initialise_game()

    def __initialise_game(self):
        # A full 52 card deck, with 4 suits and 13 ranks (A, 2-10, J, Q, K); suit is not relevant for blackjack
        self.deck = ["A", "A", "A", "A", 
                     2, 2, 2, 2, 
                     3, 3, 3, 3,
                     4, 4, 4, 4,
                     5, 5, 5, 5,
                     6, 6, 6, 6,
                     7, 7, 7, 7,
                     8, 8, 8, 8,
                     9, 9, 9, 9,
                     10, 10, 10, 10,
                     "J", "J", "J", "J",
                     "Q", "Q", "Q", "Q",
                     "K", "K", "K", "K"]
        random.shuffle(self.deck)
        self.player_hand = [self.__draw_card(), self.__draw_card()]
        self.dealer_hand = [self.__draw_card()]

    def __draw_card(self):
        if len(self.deck) == 0:
            raise Exception("No more cards in the deck.")
        return self.deck.pop()
    
    def __hand_value(self, hand):
        value = 0
        aces = 0
        usable_ace = False 
        # A usable ace is an ace that can be counted as 11 without busting the hand. 
        # We track this to determine if the player has a "soft" hand (one that includes an ace counted as 11) or 
        # a "hard" hand (one where all aces are counted as 1).

        # Calculate the value of the hand adding aces as 1 for now
        for card in hand:
            if card in ["J", "Q", "K"]:
                value += 10
            elif card == "A":
                aces += 1 # Count aces separately to decide later if they should be 1 or 11
                value += 1
            else:
                value += card
        
        # Now decide if we can treat any aces as 11 without exceeding 21
        if aces > 0 and value + 10 <= 21:
            usable_ace = True # We have at least one ace that can be treated as 11
            value += 10
        return value, usable_ace
    
    # Note: The RL policy is only used when the player's hand value is between 12 and 20.
    # If the value is less than 12, the player must HIT; if it is 21, the player must STAND.
    # The exceptions are raised to enforce these rules and prevent invalid actions when training the RL agent.
    def __hit(self, hand):
        #Should this send an immediate reward of 0 and then when agent decides to stand it waits until outcome is sent?
        value, _ = self.__hand_value(hand)

        if hand == self.player_hand and value >= 21:
            print("Player cannot hit if hand value is 21 or more.")
            self.__outcome()
        if hand == self.dealer_hand and value >= 17:
            print("Dealer cannot hit if hand value is 17 or more.")
            return hand
        
        card = self.__draw_card()
        print("Drew card:", card)
        hand.append(card)
        return hand
    
    def __stand(self):
        value, _ = self.__hand_value(self.player_hand)

        if value < 12:
            print("Player must hit if hand value is less than 12.")
            return self.player_hand
        
        self.__dealer_play() # After the player stands, the dealer will play

    def step(self, action):
        if action == 'hit':
            self.__hit(self.player_hand)
            reward = 0
            done = False
            # Check if player exceeds 21 after hitting
            player_value, _ = self.__hand_value(self.player_hand)
            if player_value > 21:
                reward = -1
                done = True
                self.__outcome() # Determine outcome immediately if player exceeds 21
        elif action == 'stand':
            self.__stand()
            reward, done = self.outcome # Outcome is determined after dealer plays
        else:
            raise ValueError("Invalid action. Action must be 'hit' or 'stand'.")
        
        return self.get_state(), reward, done

    def __dealer_play(self):
        print("Dealer plays...")
        while self.__hand_value(self.dealer_hand)[0] < 17: 
            self.__hit(self.dealer_hand)
            
        print("Dealer's hand:", self.dealer_hand)

        self.__outcome() # After the dealer finishes playing, we determine the outcome of the game

    def __outcome(self):
        player_value = self.__hand_value(self.player_hand)[0]
        dealer_value = self.__hand_value(self.dealer_hand)[0]

        # Note: The flow of if statements is important here. We check for player bust first, then dealer bust, then compare values.
        # This ensures we correctly identify the outcome of the game based on the rules of blackjack.
        if player_value > 21:
            print("Player loses (Exceeded 21), dealer wins.")
            self.outcome = -1
        elif dealer_value > 21:
            print("Dealer loses (Exceeded 21), player wins.")
            self.outcome = 1
        elif player_value > dealer_value:
            print("Player wins.")
            self.outcome = 1
        elif dealer_value > player_value:
            print("Dealer wins.")
            self.outcome = -1
        elif player_value == dealer_value:
            print("Draw.")
            self.outcome = 0

    def get_state(self): # Returns the RL state
        player_value, usable_ace = self.__hand_value(self.player_hand)
        dealer_card = self.dealer_hand[0] # Dealers visible card

        if dealer_card in ["J", "Q", "K"]:
            dealer_card = 10
        elif dealer_card == "A":
            dealer_card = 11
        return player_value, dealer_card, usable_ace


'''
Reward structure clarification for Blackjack RL environment

Suggestion: The environment should return reward = 0 after every HIT action,
and only return a non-zero reward when the player chooses STAND (i.e. at the
terminal transition after the dealer finishes playing).

Expected behaviour:

1) After HIT:
    - Environment draws a card for the player
    - Returns next player state
    - reward = 0
    - done = False (unless player busts → then reward = -1, done = True)

2) After STAND:
    - Environment runs dealer policy internally until dealer stops
    - Compare dealer vs player totals
    - reward = +1 (win), 0 (draw), or -1 (loss)
    - done = True

Important reasoning:
The dealer's actions are environment dynamics, not agent decisions, so they
should NOT appear as intermediate states in the agent’s trajectory. From the
agent’s perspective:
    (state, STAND) → terminal reward

This structure is required so Monte-Carlo, SARSA, and Q-learning update rules
work correctly, since Blackjack is a terminal-reward episodic environment where
intermediate rewards are zero until the outcome is determined.
'''


'''
def step(self, action):
        if action not in ["hit", "stick"]:
            raise ValueError("Action must be 'hit' or 'stick'.")

        if action == "hit":
            self.hit(self.player_hand)
            player_value, _ = self.__hand_value(self.player_hand)[0]

            if player_value > 21:
                self.__outcome()
                return None, -1, True

            return self.get_state(), 0, False

        if action == "stick":
            self.stand()
            reward = self.reward
            return None, reward, True
'''