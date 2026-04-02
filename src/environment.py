import random

class Environment:
    def __init__(self):
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []
        self.initialise_game()

    def initialise_game(self):
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
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card()]

    def draw_card(self):
        if len(self.deck) == 0:
            raise Exception("No more cards in the deck.")
        return self.deck.pop()
    
    def hand_value(self, hand):
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
    def hit(self, hand):
        value, _ = self.hand_value(hand)

        if hand == self.player_hand and value >= 21:
            raise Exception("Player cannot hit if hand value is 21 or more.")
        if hand == self.dealer_hand and value >= 17:
            raise Exception("Dealer must stand if hand value is 17 or more.")
        
        card = self.draw_card()
        hand.append(card)
        return hand
    
    def player_stand_and_dealer_plays(self):
        value, _ = self.hand_value(self.player_hand)

        if value < 12:
            raise Exception("Player must hit if hand value is less than 12.")
        
        self.dealer_play() # After the player stands, the dealer will play
        
    def dealer_play(self):
        while self.hand_value(self.dealer_hand)[0] < 17: 
            self.hit(self.dealer_hand)

        self.outcome() # After the dealer finishes playing, we determine the outcome of the game
    
    def outcome(self):
        player_value = self.hand_value(self.player_hand)[0]
        dealer_value = self.hand_value(self.dealer_hand)[0]

        # Note: The flow of if statements is important here. We check for player bust first, then dealer bust, then compare values.
        # This ensures we correctly identify the outcome of the game based on the rules of blackjack.
        if player_value > 21:
            return "Player loses, dealer wins.", -1
        elif dealer_value > 21:            
            return "Dealer loses, player wins.", 1
        elif player_value > dealer_value:
            return "Player wins.", 1
        elif dealer_value > player_value:
            return "Dealer wins.", -1
        elif player_value == dealer_value:
            return "Draw.", 0
        
    def get_state(self): # Returns the RL state
        player_value, usable_ace = self.hand_value(self.player_hand)
        dealer_card = self.dealer_hand[0] # The dealer's visible card
        
        if dealer_card in ["J", "Q", "K"]:
            dealer_card = 10
        elif dealer_card == "A":
            dealer_card = 11
        return player_value, dealer_card, usable_ace