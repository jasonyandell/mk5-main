# Texas 42 Rules in Gherkin Format

## Equipment and Setup

### Feature: Game Setup
  Background:
    Given a standard double-six domino set with 28 pieces
    And exactly 4 players
    And players are arranged in 2 partnerships
    And partners sit opposite each other

  [ ] Scenario: Determining First Dealer
    Given all four players are ready to start
    When each player draws one domino face-down
    Then the player with the highest total pip count becomes the first shaker
    And if there is a tie, the affected players must redraw

  [ ] Scenario: Drawing Dominoes (Tournament Standard)
    Given the shaker has shuffled dominoes face-down
    When players draw dominoes
    Then the non-shaking team draws first with 7 dominoes each
    And the shaker's partner draws next with 7 dominoes
    And the shaker draws last with 7 dominoes
    And no dominoes remain

  [ ] Scenario: Domino Arrangement
    Given players have drawn their dominoes
    When arranging dominoes for tournament play
    Then dominoes must be arranged in 4-3 or 3-4 formation
    And once bidding begins, dominoes cannot be rearranged

## Bidding Rules

### Feature: Standard Bidding

  [ ] Scenario: Bidding Order
    Given a new hand has been dealt
    When bidding begins
    Then the player to the left of the shaker bids first
    And bidding proceeds clockwise
    And each player gets exactly one opportunity to bid or pass

  [ ] Scenario: Valid Point Bids
    Given it is a player's turn to bid
    When they make a point bid
    Then valid bids are 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, or 41 points

  [ ] Scenario: Valid Mark Bids
    Given it is a player's turn to bid
    When they make a mark bid
    Then 1 mark equals 42 points
    And 2 marks equals 84 points
    And higher marks equal multiples of 42 points

  [ ] Scenario: Opening Bid Constraints
    Given no bids have been made
    When a player makes the opening bid
    Then the minimum bid is 30 points
    And the maximum opening bid is 2 marks (84 points)
    And the exception is a plunge bid which may open at 4 marks

  [ ] Scenario: Sequential Bidding
    Given at least one bid has been made
    When a player bids
    Then their bid must exceed the previous bid
    And after reaching 42 (1 mark), subsequent bids must be in mark increments
    And any player may bid up to 2 marks when 2 marks has not already been bid
    And subsequent bids after 2 marks may only be one additional mark

  [ ] Scenario: Three Mark Bidding
    Given the current bid is less than 3 marks
    When a player wants to bid 3 marks
    Then they can only bid 3 marks if another player has already bid 2 marks
    And 3 marks cannot be used as an opening bid under tournament rules

### Feature: Special Bids

  [ ] Scenario: Plunge Bid Requirements
    Given a player holds at least 4 doubles in their hand
    When they want to plunge
    Then they must bid at least 4 marks
    And if bidding has reached 4 marks, they must bid 5 marks
    And this can be declared as an opening bid or jump bid
    And this is the only case where jump bidding is allowed

  [ ] Scenario: Plunge Bid Mechanics
    Given a player has successfully bid a plunge
    When play begins
    Then their partner names trump without consultation
    And their partner leads the first trick
    And they must win all 7 tricks to succeed

  [ ] Scenario: All Players Pass
    Given all players have had a chance to bid
    When all players pass
    Then under tournament rules, the hand is reshaken with the next player as shaker
    And under common variation, the shaker must bid minimum 30

## Gameplay Mechanics

### Feature: Trump Declaration

  [ ] Scenario: Declaring Trump
    Given a player has won the bidding
    When they are ready to play
    Then they must declare trump before playing the first domino
    And trump options include any suit (blanks through sixes)
    And trump options include doubles as trump
    And trump options include no-trump (follow-me)

### Feature: Playing Tricks

  [ ] Scenario: Leading a Trick
    Given it is time to play a trick
    When determining who leads
    Then the bid winner leads to the first trick
    And the winner of each trick leads to the next trick
    And any domino may be led

  [ ] Scenario: Following Suit
    Given a domino has been led
    And it is not a trump
    When determining the suit
    Then the higher end of the domino determines the suit led
    And players must play a domino of the led suit if possible
    And if unable to follow suit, players may play trump
    And if unable to follow suit or trump, players may play any domino

  [ ] Scenario: Winning a Trick
    Given all players have played to a trick
    When determining the winner
    Then the highest trump played wins
    And if no trump was played, the highest domino of the led suit wins
    And in special games like Sevens, the first played wins ties

### Feature: Doubles Treatment

  [ ] Scenario: Standard Doubles Rules
    Given standard tournament rules apply
    When playing with doubles
    Then doubles belong to their natural suit
    And 6-6 is the highest six
    And 5-5 is the highest five
    And when doubles are trump, only the seven doubles are trump

  [ ] Scenario: Renege
    Given a player has failed to follow suit when able
    When a renege is detected
    Then in tournament play, it results in immediate loss of hand plus penalty marks
    And in casual play, it often results in just loss of hand
    And it may be called when noticed and verified by examining played dominoes

## Scoring Systems

### Feature: Point Values

  [ ] Scenario: Counting Dominoes
    Given dominoes are being scored
    When calculating point values
    Then 5-5 is worth 10 points
    And 6-4 is worth 10 points
    And 5-0 is worth 5 points
    And 4-1 is worth 5 points
    And 3-2 is worth 5 points
    And all other dominoes are worth 0 points
    And the total count value is 35 points

  [ ] Scenario: Trick Points
    Given tricks are being scored
    When calculating trick points
    Then each trick won is worth 1 point
    And there are 7 total tricks worth 7 points
    And the hand total is 35 (count) + 7 (tricks) = 42 points

### Feature: Mark System Scoring

  [ ] Scenario: Successful Bids
    Given a team has made their bid
    When awarding marks
    Then a successful 30-41 point bid earns 1 mark
    And a successful 1 mark bid (42 points) earns 1 mark
    And a successful 2 mark bid earns 2 marks
    And higher bids earn marks equal to the bid

  [ ] Scenario: Failed Bids
    Given a team has failed to make their bid
    When awarding marks
    Then the opponents receive marks equal to what was bid

## Victory Conditions

### Feature: Game Victory

  [ ] Scenario: Mark System Victory
    Given teams are playing with the mark system
    When checking for game victory
    Then the first partnership to accumulate 7 marks wins

  [ ] Scenario: Point System Victory
    Given teams are playing with the point system
    When checking for game victory
    Then the first partnership to reach the target score wins
    And the target is usually 250 points

### Feature: Hand Victory

  [ ] Scenario: Bidding Team Wins
    Given a hand has been played
    When the bidding team takes points equal to or exceeding their bid
    Then the bidding team wins the hand

  [ ] Scenario: Defending Team Wins
    Given a hand has been played
    When the bidding team fails to take enough points
    Then the defending team wins by "setting" the bidders

## Special Contracts

### Feature: Nel-O Contract

  [ ] Scenario: Nel-O Requirements
    Given a player wants to bid Nel-O
    When making the bid
    Then they must bid at least 1 mark
    And their objective is to lose every trick

  [ ] Scenario: Nel-O Gameplay
    Given Nel-O has been bid
    When playing the hand
    Then the bidder's partner sits out with dominoes face-down
    And no trump suit is declared
    And doubles may form their own suit (standard)
    And doubles may remain high in suits (variation)
    And doubles may become low in suits (variation)

### Feature: Special Bid Restrictions

  [ ] Scenario: Tournament Restrictions
    Given a tournament game is being played
    When players are bidding
    Then Nel-O is not allowed
    And Plunge is not allowed unless holding 4+ doubles
    And Splash is not allowed
    And Sevens is not allowed

## Player Conduct

### Feature: Communication Rules

  [ ] Scenario: Prohibited Bidding Communication
    Given players are in the bidding phase
    When making bids
    Then no voice inflection may be used to signal hand strength
    And no gestures or physical signals are allowed
    And no commentary beyond bid declaration is permitted
    And no hesitation for strategic effect is allowed

  [ ] Scenario: Prohibited Play Communication
    Given players are in the play phase
    When playing dominoes
    Then no verbal communication about game state is allowed
    And no tapping, positioning, or gesturing is permitted
    And players cannot announce trump, count, or strategy
    And timing cannot be used to convey information

### Feature: Tournament Conduct

  [ ] Scenario: Domino Handling
    Given a tournament game is in progress
    When handling dominoes
    Then the first domino touched must be played
    And exposed dominoes must be played at first legal opportunity
    And players cannot rearrange their hand after bidding begins
    And players must follow official's instructions

  [ ] Scenario: Violation Penalties
    Given a player has violated communication rules
    When penalties are assessed
    Then first offense results in a warning
    And second offense results in a mark to opponents
    And severe violations result in ejection from game/tournament