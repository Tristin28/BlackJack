import random
#Should we use same object but have a method which will reset it to keep same object instead of having to create a new one for each episode?
class Environment:
    def __init__(self):
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []
        self.reward = 0
        self.done = False
        self.__initialise_game()

    def __initialise_game(self):
        self.reward = 0
        self.done = False
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
        self.advance_to_learning_state()

    def advance_to_learning_state(self):
            while True:
                player_value, _ = self.__hand_value(self.player_hand)

                if player_value > 21:
                    self.__outcome()
                    return None

                if player_value == 21:
                    self.__dealer_play()
                    return None

                if player_value < 12:
                    self.__hit(self.player_hand)
                    continue

                return self.get_state()

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
        card = self.__draw_card()
        hand.append(card)
    
    def __stand(self):
        value, _ = self.__hand_value(self.player_hand)

        if value < 12:
            raise Exception("Player shouldn't be able to stand with hand value less than 12.")
        
        self.__dealer_play() # After the player stands, the dealer will play

    def step(self, action):
        value, _ = self.__hand_value(self.player_hand)

        if self.done:
            raise Exception("Episode has ended. Step shouldn't have been called.")
        
        if action == 'HIT':
            if value >= 21:
                raise Exception("Player shouldn't be able to hit with hand value of 21 or more.")
            
            self.__hit(self.player_hand)

            next_state = self.advance_to_learning_state()
                
            return next_state, self.reward, self.done
        
        elif action == 'STAND':
            self.__stand()
            return None, self.reward, self.done
        else:
            raise ValueError("Invalid action. Action must be 'HIT' or 'STAND'.")
    
    def __dealer_play(self):
        while self.__hand_value(self.dealer_hand)[0] < 17: 
            self.__hit(self.dealer_hand)
            
        self.__outcome() # After the dealer finishes playing, we determine the outcome of the game

    def __outcome(self):
        self.done = True
        player_value, _ = self.__hand_value(self.player_hand)
        dealer_value, _ = self.__hand_value(self.dealer_hand)

        # Note: The flow of if statements is important here. We check for player bust first, then dealer bust, then compare values.
        # This ensures we correctly identify the outcome of the game based on the rules of blackjack.
        if player_value > 21: # Player loses (Exceeded 21), dealer wins.
            self.reward = -1
        elif dealer_value > 21: # Dealer loses (Exceeded 21), player wins.
            self.reward = 1
        elif player_value > dealer_value: # Player wins.
            self.reward = 1
        elif dealer_value > player_value: # Dealer wins.
            self.reward = -1
        elif player_value == dealer_value: # Draw.
            self.reward = 0
    
    def __dealer_visible_card_value(self):
        dealer_card = self.dealer_hand[0]
        if dealer_card in ["J", "Q", "K"]:
            return 10
        if dealer_card == "A":
            return 11
        return dealer_card

    def get_state(self):
        player_value, usable_ace = self.__hand_value(self.player_hand)
        dealer_card = self.__dealer_visible_card_value()

        return player_value, dealer_card, usable_ace