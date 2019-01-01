
from games.leduc import Leduc
from rlpoker.extensive_game import ExtensiveGame

if __name__ == "__main__":
    root = Leduc.create_tree((10, 10, 11, 11, 12, 12))

    game = ExtensiveGame(root)

    game.print_tree()