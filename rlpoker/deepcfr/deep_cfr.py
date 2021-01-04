"""This file implements Deep CFR, as described in Brown et al. - Deep Counterfactual Regret Minimization (2019).
"""
import abc
import collections
import os
import time
from functools import lru_cache
from typing import Any, Dict, NamedTuple, Sequence, List, Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import rlpoker.cfr_util
from rlpoker import best_response
from rlpoker import buffer
from rlpoker import extensive_game
from rlpoker import neural_game
from rlpoker import util
from rlpoker.cfr_game import get_available_actions, sample_chance_action, is_terminal, payoffs, which_player
from rlpoker.extensive_game import ActionFloat
from rlpoker.util import sample_action, ExperimentSummaryWriter


def check_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Make sure that x is a numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


class StrategyMemoryElement(NamedTuple):
    info_set_id: str
    t: int
    info_set_strategy: ActionFloat


class AdvantageMemoryElement(NamedTuple):
    info_set_id: str
    t: int
    info_set_advantages: Dict[Any, float]


class AdvantageBatch(abc.ABC):

    @property
    @abc.abstractmethod
    def info_set_vectors(self):
        ...

    @property
    @abc.abstractmethod
    def times(self):
        ...

    @property
    @abc.abstractmethod
    def info_set_advantages(self):
        ...


class TorchAdvantageBatch:
    def __init__(self, batch: Sequence[AdvantageMemoryElement],
                 action_indexer: neural_game.ActionIndexer,
                 info_set_vectoriser: neural_game.InfoSetVectoriser,
                 device: torch.device):
        """

        Args:
            batch: sequence of AdvantageMemoryElements.
            action_indexer: ActionIndexer. Turns actions into indices.
            info_set_vectoriser: InfoSetVectoriser. Turns info set ids into vectors.
            device: the device to put the parameters on.
        """
        self._batch = batch
        self._info_set_vectors = torch.as_tensor(
            data=[info_set_vectoriser.get_vector(element.info_set_id) for element in batch],
            dtype=torch.float32,
            device=device)
        self._times = torch.as_tensor(data=[element.t for element in batch],
                                      dtype=torch.float32, device=device).view(-1, 1)
        self._info_set_advantages = torch.as_tensor(
            data=[info_set_advantages_to_vector(action_indexer, element.info_set_advantages) for element in batch],
            dtype=torch.float32,
            device=device).view(-1, action_indexer.action_dim)

    @property
    def info_set_vectors(self):
        return self._info_set_vectors

    @property
    def times(self):
        return self._times

    @property
    def info_set_advantages(self):
        return self._info_set_advantages


class AdvantageNetwork:
    """
    A RegretPredictor can be used inside cfr_traverse.
    """

    def predict_advantages(self, info_set_vector, action_indexer: neural_game.ActionIndexer) -> \
            extensive_game.ActionFloat:
        """
        Predicts advantages for each action available in the information set.

        Args:
            info_set_vector: ndarray.
            action_indexer: ActionIndexer. The mapping from actions to indices.

        Returns:
            ActionFloat. The predicted regret for each action in the information set.
        """
        raise NotImplementedError("Not implemented in the base class.")

    def compute_action_probs(self, info_set_vector, action_indexer: neural_game.ActionIndexer):
        """
        Compute action probabilities in this information set.

        Args:
            info_set_vector: ndarray.
            action_indexer: ActionIndexer.

        Returns:
            ActionFloat. The action probabilities in this information set.
        """
        action_advantages = self.predict_advantages(info_set_vector, action_indexer)
        return rlpoker.cfr_util.compute_regret_matching(action_advantages, highest_regret=True)


class DeepAdvantageNetwork(AdvantageNetwork, nn.Module):

    def __init__(self, state_shape: Sequence[int], action_indexer: neural_game.ActionIndexer, player: int,
                 layer_sizes: Sequence[int] = (64, 64)):
        """
        A DeepAdvantageNetwork uses a neural network to predict advantages for actions in information sets.

        Args:
            state_shape: tuple of ints. The dimensions of the state vector.
            action_indexer: ActionIndexer.
            player: int. The player number it represents. Used for scoping.
        """
        super().__init__()

        self._state_shape = state_shape
        self.action_indexer = action_indexer
        self.player = player

        self.state_dim = np.prod(self._state_shape)

        self._layers = []
        prev_layer_dim = np.prod(state_shape)
        for i, layer_dim in enumerate(layer_sizes):
            self._layers.append(nn.Linear(prev_layer_dim, layer_dim))
            self._layers.append(nn.ReLU())
            prev_layer_dim = layer_dim

        self._layers.append(nn.Linear(prev_layer_dim, self.action_indexer.action_dim))
        self._layers = nn.ModuleList(self._layers)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.state_dim)
        for layer in self._layers:
            x = layer(x)

        return x

    def predict_advantages(self, info_set_vector, action_indexer: neural_game.ActionIndexer) -> \
            extensive_game.ActionFloat:

        advantages = self(torch.tensor([info_set_vector]).to(self.device).to(torch.float32))
        assert advantages.shape[0] == 1
        advantages = check_numpy(advantages)

        return extensive_game.ActionFloat({
            action: advantages[0, self.action_indexer.get_index(action)] for action in self.action_indexer.actions
        })

    @property
    @lru_cache
    def device(self):
        return next(self.parameters()).device


def info_set_advantages_to_vector(action_indexer: neural_game.ActionIndexer,
                                  info_set_advantages: Dict[Any, float]):
    """

    Args:
        action_indexer:
        info_set_advantages: dict mapping actions to advantages.

    Returns:
        vector with the advantage for each action in the correct index.
    """
    advantages = np.zeros(action_indexer.action_dim)
    for action, advantage in info_set_advantages.items():
        advantages[action_indexer.get_index(action)] = advantage

    return advantages


def early_stopping(losses: List[float], consecutive_increases: int = 2):
    """Returns True if and only if losses[-consecutive_increases-1:] is monotonically increasing.

    Args:
        losses: list of floats. The losses.
        consecutive_increases: int. The number of consecutive increases to see before early stopping.

    Returns:
        early_stop: bool. True if and only if we should early stop.
    """
    # Can't early stop before we see enough losses.
    if len(losses) <= consecutive_increases:
        return False

    relevant_losses = losses[-consecutive_increases - 1:]
    return sorted(relevant_losses) == relevant_losses


def early_stopping_water_mark(losses: List[float], num_attempts: int = 5):
    """Returns True if and only if the loss has failed to beat the low water mark in num_attempts 
    attempts.

    Args:
        losses: list of floats. The losses.
        num_attempts: int. The number of attempts to beat the low water mark.

    Returns:
        early_stop: bool. True if and only if we should early stop.
    """
    # Can't early stop before we see enough losses.
    if len(losses) <= num_attempts:
        return False

    return min(losses[-num_attempts:]) > min(losses)


class Trainer:

    def __init__(self,
                 name: str,
                 network: DeepAdvantageNetwork,
                 advantage_memory: buffer.Reservoir,
                 action_indexer: neural_game.ActionIndexer,
                 info_set_vectoriser: neural_game.InfoSetVectoriser,
                 writer: ExperimentSummaryWriter,
                 device: torch.device,
                 lr: float = 1e-3,
                 batch_size=1024):
        self._name = name
        self._network = network
        self._advantage_memory = advantage_memory
        self._action_indexer = action_indexer
        self._info_set_vectoriser = info_set_vectoriser
        self._writer = writer
        self._device = device
        self._lr = lr
        self._batch_size = batch_size
        self._opt = torch.optim.Adam(self._network.parameters(), lr=self._lr)

    def train(self, current_time: int, num_steps: int = 4000, writer: ExperimentSummaryWriter = None,
              global_step: int = 0):
        indices = list(range(len(self._advantage_memory)))
        losses = []
        for i in range(num_steps):
            # Shuffle the advantage memory.
            batch_indices = np.random.choice(indices, self._batch_size, replace=True)
            batch = self._advantage_memory.get_elements(batch_indices)

            torch_batch = TorchAdvantageBatch(batch,
                                              action_indexer=self._action_indexer,
                                              info_set_vectoriser=self._info_set_vectoriser,
                                              device=self._device)

            self._opt.zero_grad()
            loss = self._compute_loss(torch_batch, current_time)
            loss.backward()
            self._opt.step()

            losses.append(check_numpy(loss).item())

            if writer is not None:
                writer.add_scalar(f'{self._name}/loss', check_numpy(loss), global_step + i)

            # Early stopping.
            if early_stopping_water_mark(losses, num_attempts=20):
                print("Losses: {}".format(losses))
                print("Early stopping.")
                break

        return losses

    def _compute_loss(self, batch: TorchAdvantageBatch, current_time: int) -> torch.Tensor:
        """Computes the loss on one batch.

        Args:
            batch: TorchAdvantageBatch.
            current_time: int. The current iteration we are training on.

        Returns:
            loss: float.
        """
        advantages = self._network(batch.info_set_vectors)
        regrets = torch.sum((batch.info_set_advantages - advantages) ** 2, dim=1)
        return torch.mean(batch.times * regrets) / current_time


def compute_mean_strategy(strategy_memory: buffer.Reservoir):
    """Returns the mean strategy for each information set, weighted by time.

    Args:
        strategy_memory: Reservoir consisting of StrategyMemoryElement objects.

    Returns:
        Strategy.
    """
    strategies = collections.defaultdict(list)
    for info_set_id, t, info_set_strategy in strategy_memory.buffer:
        strategies[info_set_id].append((t, info_set_strategy))

    return extensive_game.compute_weighted_strategy(strategies)


def deep_cfr(
        exp_name: str,
        neural_game: neural_game.NeuralGame,
        num_iters: int = 100, num_traversals: int = 10000,
        advantage_maxlen: int = 1000000, strategy_maxlen: int = 1000000,
        batch_size: int = 1024, num_sgd_updates: int = 100):
    """
    Args:
        neural_game: NeuralGame.
        num_iters: int. The number of iterations to run deep CFR for.
        num_traversals: int. The number of traversals per CFR iteration.
        advantage_maxlen: int. The maximum length of the advantage memories.
        strategy_maxlen: int. The maximum length of the strategy memory.
        batch_size: int. The batch size to use in training.
        num_sgd_updates: int. The number of sgd updates per training.

    Returns:
        strategy, exploitability.
    """
    game, action_indexer, info_set_vectoriser = neural_game

    advantage_memory1 = buffer.Reservoir(maxlen=advantage_maxlen)
    advantage_memory2 = buffer.Reservoir(maxlen=advantage_maxlen)
    strategy_memory = buffer.Reservoir(maxlen=strategy_maxlen)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    network1 = DeepAdvantageNetwork(info_set_vectoriser.state_shape, action_indexer, 1).to(device)
    network2 = DeepAdvantageNetwork(info_set_vectoriser.state_shape, action_indexer, 2).to(device)

    writer = ExperimentSummaryWriter(exp_name=exp_name, flush_secs=20)

    # Iterate over players and do cfr traversals.
    global_step1 = 0
    global_step2 = 0
    for t in range(1, num_iters + 1):
        print("Iteration t = {}".format(t))
        for player in [1, 2]:
            print("Player: {}".format(player))
            print("Traversing")
            for i in tqdm(range(num_traversals)):
                cfr_traverse(game, action_indexer, info_set_vectoriser,
                             game.root, player, network1, network2,
                             advantage_memory1, advantage_memory2,
                             strategy_memory, t)

            # Train the traversing player's network on the cfr traversals.
            if player == 1:
                network1 = DeepAdvantageNetwork(info_set_vectoriser.state_shape, action_indexer, 1).to(device)
                trainer = Trainer('network1', network1, advantage_memory1, action_indexer,
                                  info_set_vectoriser, writer, device=device)
                global_step = global_step1
            else:
                network2 = DeepAdvantageNetwork(info_set_vectoriser.state_shape, action_indexer, 1).to(device)
                trainer = Trainer('network2', network1, advantage_memory1, action_indexer,
                                  info_set_vectoriser, writer, device=device)
                global_step = global_step2

            trainer.train(current_time=t, num_steps=4000, writer=writer, global_step=global_step)
            if player == 1:
                global_step1 += 4000
            else:
                global_step2 += 4000

        # print("################")
        #
        # print("----------------")
        # print("Advantage memory 1:")
        # print(advantage_memory1.buffer)
        # print("----------------")
        # print("Advantage memory 2:")
        # print(advantage_memory2.buffer)
        # print("----------------")
        #
        # print("################")
        #

        # print("----------------")
        # print("Predicted advantages:")
        # for info_set_id in set(game.info_set_ids.values()):
        #     print("{}: {}".format(
        #         info_set_id,
        #         network.predict_advantages(info_set_vectoriser.get_vector(info_set_id), action_indexer))
        #     )
        # print("----------------")
        #

        print("Advantage memory 1 length: {}".format(len(advantage_memory1)))
        print("Advantage memory 2 length: {}".format(len(advantage_memory2)))
        print("Strategy memory length: {}".format(len(strategy_memory)))

        print(f"Computing mean strategy.")
        mean_strategy = compute_mean_strategy(strategy_memory)
        # print("Strategy summary")
        # print(mean_strategy)
        if game.is_strategy_complete(mean_strategy):
            exploitability = best_response.compute_exploitability(game, mean_strategy)
        else:
            print("Strategy not complete, filling uniformly.")
            exploitability = best_response.compute_exploitability(
                game,
                mean_strategy,
            )
        print("Exploitability: {} mbb/h".format(exploitability * 1000))

        writer.add_scalar('exploitability_mbb_h', exploitability * 1000, global_step=t)

    # TODO(chrisn). Train the network on the strategy memory.
    return mean_strategy, exploitability


def cfr_traverse(game: extensive_game.ExtensiveGame, action_indexer: neural_game.ActionIndexer,
                 info_set_vectoriser: neural_game.InfoSetVectoriser,
                 node: extensive_game.ExtensiveGameNode, player: int,
                 network1: AdvantageNetwork, network2: AdvantageNetwork,
                 advantage_memory1: buffer.Reservoir, advantage_memory2: buffer.Reservoir,
                 strategy_memory: buffer.Reservoir, t: int):
    """

    Args:
        game: ExtensiveGame.
        action_indexer: ActionIndexer. This maps actions to indices, so that we can use neural networks.
        info_set_vectoriser: InfoSetVectoriser. This maps information sets to vectors, so we can use neural networks.
        node: ExtensiveGameNode. The current node.
        player: int. The traversing player. Either 1 or 2.
        network1: RegretPredictor. The network for player 1.
        network2: RegretPredictor. The network for player 2.
        advantage_memory1: Reservoir. The advantage memory for player 1.
        advantage_memory2: Reservoir. The advantage memory for player 2.
        strategy_memory: Reservoir. The strategy memory (for both players).
        t: int. The current iteration of deep cfr.

    Returns:

    """
    if is_terminal(node):
        return payoffs(node)[player]
    elif which_player(node) == 0:
        # Chance player
        a = sample_chance_action(node)
        return cfr_traverse(game, action_indexer, info_set_vectoriser, node.children[a], player,
                            network1, network2,
                            advantage_memory1, advantage_memory2, strategy_memory, t)
    elif which_player(node) == player:
        # It's the traversing player's turn.
        state_vector = info_set_vectoriser.get_vector(game.get_info_set_id(node))
        values = dict()
        for action in get_available_actions(node):
            child = node.children[action]
            values[action] = cfr_traverse(game, action_indexer, info_set_vectoriser, child, player,
                                          network1, network2,
                                          advantage_memory1, advantage_memory2, strategy_memory, t)
            assert values[action] is not None, print("Shouldn't be None! node was: {}".format(node))
        info_set_regrets = dict()

        # Compute the player's strategy
        network = network1 if player == 1 else network2
        if t == 1:
            # This is the equivalent of initialising the network so it starts with all zeroes.
            info_set_strategy = extensive_game.ActionFloat.initialise_uniform(action_indexer.actions)
        else:
            info_set_strategy = network.compute_action_probs(state_vector, action_indexer)

        sampled_counterfactual_value = sum([info_set_strategy[action] * values[action] for action in
                                            get_available_actions(
                                                node)])
        for action in get_available_actions(node):
            info_set_regrets[action] = values[action] - sampled_counterfactual_value

        info_set_id = game.info_set_ids[node]
        advantage_memory = advantage_memory1 if player == 1 else advantage_memory2
        advantage_memory.append(AdvantageMemoryElement(info_set_id, t, info_set_regrets))

        # In traverser infosets, the value passed back up is the weighted average of all action values,
        # where action a’s weight is info_set_strategy[a]
        return sampled_counterfactual_value
    else:
        # It's the other player's turn.
        state_vector = info_set_vectoriser.get_vector(game.get_info_set_id(node))

        # Compute the other player's strategy
        other_player = 1 if player == 2 else 2
        network = network1 if other_player == 1 else network2
        if t == 1:
            # This is the equivalent of initialising the network so it starts with all zeroes.
            info_set_strategy = extensive_game.ActionFloat.initialise_uniform(action_indexer.actions)
        else:
            info_set_strategy = network.compute_action_probs(state_vector, action_indexer)

        info_set_id = game.info_set_ids[node]
        strategy_memory.append(StrategyMemoryElement(info_set_id, t, info_set_strategy))

        action = sample_action(info_set_strategy, available_actions=get_available_actions(node))
        return cfr_traverse(game, action_indexer, info_set_vectoriser, node.children[action], player,
                            network1, network2, advantage_memory1, advantage_memory2, strategy_memory, t)
