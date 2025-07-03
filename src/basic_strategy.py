def basic_strategy(player_cards, dealer_up_card):
    """
    Returns optimal Blackjack move (Hit, Stand, Double, Split) based on basic strategy.
    
    Assumed table rules:
    - 3:2 blackjack payout
    - Double after split allowed
    - Dealer hits on soft 17
    - No late surrender
    - Double on any two cards
    - 4-8 decks

    Takes as inputs:
    player_cards: list of strings of cards player currently holds
    dealer_up_card: single string of visible dealer card

    Card input strings: '2'-'10', 'J', 'Q', 'K', 'A'
    """

    # map card string values to ints
    str_to_int = {
        '2': 2, 'Two': 2,
        '3': 3, 'Three': 3,
        '4': 4, 'Four': 4,
        '5': 5, 'Five': 5,
        '6': 6, 'Six': 6,
        '7': 7, 'Seven': 7,
        '8': 8, 'Eight': 8,
        '9': 9, 'Nine': 9,
        '10': 10, 'Ten': 10,
        'J': 10, 'Jack': 10,
        'Q': 10, 'Queen': 10,
        'K': 10, 'King': 10,
        'A': 11, 'Ace': 11
    }
    
    # convert player hand and dealer card values
    player_hand = [str_to_int[card] for card in player_cards]
    dealer_card = str_to_int[dealer_up_card]

    # edge case: hand busted (> 21)
    aces = player_hand.count(11)
    total = sum(player_hand) - (10 * aces)
    if total > 21:
        return 'Bust'

    # case 1: dealt a pair (could split)
    if len(player_hand) == 2 and player_hand[0] == player_hand[1]:
        pair = player_hand[0]
        # boolean split map
        split_map = {
            2: dealer_card in range(2, 8),
            3: dealer_card in range(2, 8),
            4: dealer_card in [5, 6],
            5: False,
            6: dealer_card in range(2, 7),
            7: dealer_card in range(2, 8),
            8: True,
            9: dealer_card not in [7, 10, 11],
            10: False,
            11: True
        }
        if split_map[pair]:
            return 'Split'
    
    # case 2: soft hand (contains an ace and under 22 total)
    if 11 in player_hand and sum(player_hand) <= 21:
        soft_total = sum(player_hand)
        soft_map = {
            13: 'Double' if dealer_card in [5, 6] else 'Hit',
            14: 'Double' if dealer_card in [5, 6] else 'Hit',
            15: 'Double' if dealer_card in [4, 5, 6] else 'Hit',
            16: 'Double' if dealer_card in [4, 5, 6] else 'Hit',
            17: 'Double' if dealer_card in range(3, 7) else 'Hit',
            18: 'Double' if dealer_card in range(2, 7) else ('Stand' if dealer_card in [7, 8] else 'Hit'),
            19: 'Double' if dealer_card == 6 else 'Stand',
            20: 'Stand',
            21: 'Stand'
        }
        return soft_map[soft_total]

    # case 3: hard hand (no ace)
    hard_total = sum(player_hand)

    if hard_total <= 8:
        return 'Hit'
    if hard_total >= 17:
        return 'Stand'
    
    hard_map = {
        9: 'Double' if dealer_card in range(3, 7) else 'Hit',
        10: 'Double' if dealer_card in range(2, 10) else 'Hit',
        11: 'Double',
        12: 'Stand' if dealer_card in range(4, 7) else 'Hit',
        13: 'Stand' if dealer_card in range(2, 7) else 'Hit',
        14: 'Stand' if dealer_card in range(2, 7) else 'Hit',
        15: 'Stand' if dealer_card in range(2, 7) else 'Hit',
        16: 'Stand' if dealer_card in range(2, 7) else 'Hit'
    }
    return hard_map[hard_total]
