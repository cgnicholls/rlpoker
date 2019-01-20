
from collections import namedtuple

Card = namedtuple('Card', ['value', 'suit'])

def get_deck(num_values, num_suits):
    return [Card(value, suit) for value in range(num_values) for suit in range(num_suits)]