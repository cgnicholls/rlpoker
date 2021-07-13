import unittest

from rlpoker.games.game_builder import buildExtensiveGame


class TestExtensiveGameBuilder(unittest.TestCase):

    def test_build(self):
        buildExtensiveGame('Leduc:values_3:suits_2')
        buildExtensiveGame('OneCardPoker:values_4')
