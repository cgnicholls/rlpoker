from rlpoker.extensive_game import ExtensiveGame
from rlpoker.games.leduc import LeducBuilder
from rlpoker.games.one_card_poker import OneCardPokerBuilder
from rlpoker.games.rock_paper_scissors import RockPaperScissorsBuilder


def buildExtensiveGame(spec: str) -> ExtensiveGame:
    parts = spec.split(':')
    game = parts[0]
    game_spec = ':'.join(parts[1:])

    if game == 'Leduc':
        return LeducBuilder.build(game_spec)
    elif game == 'OneCardPoker':
        return OneCardPokerBuilder.build(game_spec)
    elif game == 'RockPaperScissors':
        return RockPaperScissorsBuilder.build(game_spec)
    else:
        raise ValueError(f"Undefined game: {game}")
