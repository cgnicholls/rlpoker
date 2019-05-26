import argparse
from collections import namedtuple
from time import gmtime, strftime
import os
import yaml

import numpy as np
import tensorflow as tf

from rlpoker.nfsp_game import NFSPGame
from rlpoker.games.leduc import LeducNFSP
from rlpoker.games.card import get_deck
from rlpoker.agent import Agent, NetSizes
from rlpoker.best_response import compute_exploitability
from rlpoker.util import TBSummariser

Hyperparameters = namedtuple('Hyperparameters',
    ['max_replay', 'max_supervised', 'best_response_lr', 'supervised_lr',
        'steps_before_training', 'eta', 'update_target_q_every',
        'initial_epsilon', 'final_epsilon', 'epsilon_steps', 'batch_size',
        'q_learn_every', 'policy_learn_every', 'clip_reward', 'net_sizes'])

def compute_epsilon(initial_epsilon, final_epsilon, train_step, epsilon_steps):
    train_fraction = float(train_step) / float(epsilon_steps)
    if train_fraction < 0.0:
        train_fraction = 0.0
    if train_fraction > 1.0:
        train_fraction = 1.0
    return (1-train_fraction) * initial_epsilon + train_fraction * final_epsilon


def compute_epsilon_sqrt_t(initial_epsilon, train_step, epsilon_steps):
    return initial_epsilon / np.sqrt(max(1, train_step / epsilon_steps))


def compute_agent_exploitability(agent: Agent, sess: tf.Session, game: NFSPGame):
    """Computes the exploitability of the agent's current strategy.

    Args:
        agent: Agent.
        sess: tensorflow session.
        game: ExtensiveGame.

    Returns:
        float. Exploitability of the agent's strategy.
    """
    states = game._state_vectors
    strategy = agent.get_strategy(sess, states)

    return compute_exploitability(game._game, strategy)


def build_transitions(states, actions, rewards):
    """Creates a dictionary with keys the players and values a list of transitions for that player. Each transition
    is a dictionary with keys 'state', 'action', 'reward', 'next_state', 'is_terminal'.
    """
    players = list(rewards.keys())
    transitions = {player: [] for player in players}
    for player in players:
        num_transitions = len(actions[player])
        assert len(states[player]) == num_transitions + 1
        for i in range(num_transitions):
            is_terminal = True if i == num_transitions - 1 else False
            reward = rewards[player] if is_terminal else 0.0
            transitions[player].append(
                {'state': states[player][i],
                 'next_state': states[player][i+1],
                 'action': actions[player][i],
                 'reward': reward,
                 'terminal': is_terminal})

    return transitions


# agents: a dictionary with keys 1, 2 and values the two agents.
def nfsp(game, hypers: Hyperparameters, train_players=(1, 2), max_train_steps=10000000, verbose=False):

    # Create two agents
    agents = {1: Agent('1', game.state_dim, game.action_dim,
                       best_response_lr=hypers.best_response_lr, supervised_lr=hypers.supervised_lr,
                       net_sizes=hypers.net_sizes, max_replay=hypers.max_replay, max_supervised=hypers.max_supervised),
              2: Agent('2', game.state_dim, game.action_dim,
                       best_response_lr=hypers.best_response_lr, supervised_lr=hypers.supervised_lr,
                       net_sizes=hypers.net_sizes, max_replay=hypers.max_replay, max_supervised=hypers.max_supervised)}

    # Create summary tensors
    summary_names = ['q_loss_1', 'q_loss_2', 'policy_loss_1', 'policy_loss_2', 'exploitability_1',
            'exploitability_2', 'epsilon']
    summariser = TBSummariser(summary_names)

    time_str = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    save_path = os.path.join('experiments', time_str)

    if not os.path.exists(save_path):
        print("Path doesn't exist, so creating: {}".format(save_path))
        os.makedirs(save_path)

    # Dump the hyperparameters
    hypers_file = os.path.join(save_path, 'hypers.yaml')
    with open(hypers_file, 'w') as f:
        print("Saving hyperparameters to {}".format(hypers_file))
        yaml.dump(hypers, f)

    with open(hypers_file, 'r') as f:
        hypers_check = yaml.load(f)
        assert hypers == hypers_check

    log_file = os.path.join(save_path, 'nfsp.log')
    print("Log file {}".format(log_file))

    with open(log_file, 'a') as f:
        f.write("Using hyperparameters: {}\n".format(hypers))

    # Create the session and initialise all variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf_train_writer = tf.summary.FileWriter(save_path, graph=sess.graph)

    print("To run tensorboard: tensorboard --logdir {}".format(os.path.join(os.getcwd(), save_path)))

    epsilon = hypers.initial_epsilon

    q_losses = {1: [], 2: []}
    policy_losses = {1: [], 2: []}

    # Update the target network to start with
    if verbose:
        with open(log_file, 'a') as f:
            f.write("Updating target networks\n")

    for agent in agents.values():
        agent.update_target_network(sess)

    # Collect rollouts from a game with the two agents.
    for train_step in range(max_train_steps):
        # Choose random player to start the game
        first_player = np.random.choice([1, 2])
        if verbose:
            with open(log_file, 'a') as f:
                f.write("First player: {}\n".format(first_player))

        states = {1: [], 2: []}
        actions = {1: [], 2: []}
        supervised = {1: [], 2: []}
        # Select the strategies
        strategy1 = np.random.choice(['q', 'policy'], p=[hypers.eta, 1.0-hypers.eta])
        strategy2 = np.random.choice(['q', 'policy'], p=[hypers.eta, 1.0-hypers.eta])
        strategies = {1: strategy1, 2: strategy2}

        if verbose:
            with open(log_file, 'a') as f:
                f.write("Strategies: {}\n".format(strategies))

        # Play one game
        next_player, state, available_actions, _, _ = game.reset(first_player)
        if verbose:
            with open(log_file, 'a') as f:
                f.write("Current node: {}\n".format(game._current_node))
        terminal = False
        player = next_player
        while not terminal:
            if verbose:
                with open(log_file, 'a') as f:
                    f.write("Player: {}\n".format(player))
                    f.write("State: {}\n".format(state))
            agent = agents[player]
            strategy = strategies[player]
            # Sample the action from the corresponding policy
            if strategy == 'q':
                if verbose:
                    with open(log_file, 'a') as f:
                        f.write("Playing with q\n")
                # Epsilon greedy strategy
                if np.random.random() < epsilon:
                    action = np.random.choice(available_actions)
                else:
                    q_values = agent.predict_q(sess, np.array([state])).ravel()
                    for i in range(len(q_values)):
                        if i not in available_actions:
                            q_values[i] = -np.inf
                    action = np.argmax(q_values)
            else:
                if verbose:
                    with open(log_file, 'a') as f:
                        f.write("Playing with policy\n")
                policy = agent.predict_policy(sess, np.array([state])).ravel()
                policy = normalise_policy(policy, available_actions)

                # We first normalise the probabilities to the available
                # actions.
                action = np.random.choice([0, 1, 2], p=policy)

            if verbose:
                with open(log_file, 'a') as f:
                    f.write("Takes action: {}\n".format(action))

            next_player, next_state, available_actions, rewards, terminal = game.step(action)

            if verbose:
                with open(log_file, 'a') as f:
                    f.write("Next player: {}\n".format(next_player))
                    f.write("Rewards: {}\n".format(rewards))
                    f.write("Current node: {}\n".format(game._current_node))

            # Add the transitions (ignoring reward and terminal for now,
            # since we will update this later).
            states[player].append(state)
            actions[player].append(action)

            # Only add to the supervised learning memory if we were playing our
            # best response strategy.
            if strategy == 'q':
                supervised[player].append({'state': state, 'action': action})

            # Set the next player and next state
            player = next_player
            state = next_state

        # This state is terminal, so add a terminal state to each player's states.
        # TODO: Make more generic. Currently next_state makes sense for both players, but this isn't the case for
        # all games.
        states[1].append(next_state)
        states[2].append(next_state)

        # Now build the transitions for each player from states, actions and rewards.
        # TODO: should we clip rewards? Hard to converge to Nash equilibrium if so, as we are altering the utilities
        # of the players.
        if hypers.clip_reward:
            rewards = {k: np.clip(v, -1.0, 1.0) for k, v in rewards.items()}

        transitions = build_transitions(states, actions, rewards)

        if verbose:
            with open(log_file, 'a') as f:
                f.write("Terminal node: {}\n".format(game._current_node))

        # The game just ended, so the last frame was terminal. The game returns
        # rewards in the order: first player, second player, so we assign them
        # to the correct agents.

        for player in [1, 2]:
            if verbose:
                with open(log_file, 'a') as f:
                    f.write("Adding transitions to player: {}\n".format(player))
                    f.write(str(transitions[player]) + '\n')
            agents[player].append_replay_memory(transitions[player])
            agents[player].append_supervised_memory(supervised[player])

        # Train the Q-networks
        if train_step >= hypers.steps_before_training:
            epsilon = compute_epsilon_sqrt_t(hypers.initial_epsilon, train_step, hypers.epsilon_steps)
            for player, agent in agents.items():
                if player not in train_players:
                    continue
                if train_step % hypers.q_learn_every == 0:
                    q_loss = agent.train_q_network(sess, hypers.batch_size)
                    q_losses[player].append(q_loss)
                if train_step % hypers.policy_learn_every == 0:
                    policy_loss = agent.train_policy_network(sess, hypers.batch_size)
                    policy_losses[player].append(policy_loss)

                # Update the target networks
                if train_step % hypers.update_target_q_every == 0:
                    if verbose:
                        with open(log_file, 'a') as f:
                            f.write("Updating target network of {}\n".format(player))
                    agent.update_target_network(sess)

        # Evaluate the best response network (the q network) for player 1
        # against player 2's average policy (the policy network), and vice
        # versa.
        if train_step % hypers.update_target_q_every == 0:
            with open(log_file, 'a') as f:
                f.write("Train step: {}\n".format(train_step))
            if train_step > hypers.steps_before_training:
                exploit1 = compute_agent_exploitability(agents[1], sess, game)
                exploit2 = compute_agent_exploitability(agents[2], sess, game)

                with open(log_file, 'a') as f:
                    f.write("Exploitabilities: {}, {}\n".format(exploit1, exploit2))
                    f.write("Q losses: {}, {}\n".format(np.mean(q_losses[1]),
                                                    np.mean(q_losses[2])))
                    f.write("Policy losses: {}, {}\n".format(
                        np.mean(policy_losses[1]),
                        np.mean(policy_losses[2])))
                    f.write("Epsilon: {}\n".format(epsilon))
                    f.write("-------------------\n")

                scalar_values = {
                    'q_loss_1': np.mean(q_losses[1]),
                    'q_loss_2': np.mean(q_losses[2]),
                    'policy_loss_1': np.mean(policy_losses[1]),
                    'policy_loss_2': np.mean(policy_losses[2]),
                    'exploitability_1': exploit1,
                    'exploitability_2': exploit2,
                    'epsilon': epsilon
                }
                print("Summarising")
                print(scalar_values)
                summary = summariser.summarise(sess, scalar_values)
                tf_train_writer.add_summary(summary, train_step)
                tf_train_writer.flush()

                q_losses = {1: [], 2: []}
                policy_losses = {1: [], 2: []}

            with open(log_file, 'a') as f:
                for i in [1, 2]:
                    f.write("Player {}, buffer sizes: RL {}, SL {}\n".format(i, len(agents[i].replay_memory),
                                                                             len(agents[i].supervised_memory)))

    return agents


def normalise_policy(policy, available_actions):
    assert len(policy.shape) == 1
    one_hot = np.zeros(policy.shape[0], dtype=float)
    for i in available_actions:
        one_hot[i] = 1.0

    policy *= one_hot

    return policy / np.sum(policy)


def sample_hypers():
    max_replay = int(np.random.choice([50000, 200000, 400000]))
    max_supervised = int(np.random.choice([200000, 400000, 1000000, 2000000]))
    best_response_lr = 10.0**(-5 + 5 * np.random.random())
    supervised_lr = 10.0**(-5 + 5 * np.random.random())
    steps_before_training = int(np.random.randint(1000, 100000))
    eta = np.random.random() * 0.6
    update_target_q_every = int(np.random.choice([200, 300, 1000]))
    epsilon_steps = int(np.random.choice([10000, 50000, 100000]))
    batch_size = int(np.random.choice([32, 64, 128, 256]))
    learn_every = int(np.random.choice([1, 8, 32, 128]))
    num_hidden = int(np.random.choice([1, 2]))
    hidden_dim = int(np.random.choice([16, 32, 64, 128]))
    hypers = Hyperparameters(max_replay=max_replay, max_supervised=max_supervised,
            best_response_lr=best_response_lr, supervised_lr=supervised_lr,
            steps_before_training=steps_before_training, eta=eta,
            update_target_q_every=update_target_q_every, initial_epsilon=0.1,
            final_epsilon=0.0, epsilon_steps=epsilon_steps, batch_size=batch_size,
            q_learn_every=learn_every, policy_learn_every=learn_every,
            clip_reward=False,
            net_sizes=NetSizes(num_hidden, hidden_dim, num_hidden, hidden_dim))

    return hypers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_values', type=int, default=3,
                        help='The number of values in the deck of cards. Default is 3.')
    parser.add_argument('--num_suits', type=int, default=2,
                        help='The number of suits in the deck of cards. Default is 2.')
    parser.add_argument('--clip_reward', action='store_true',
                        help='If given, then clip rewards to -1, 1 range.')
    parser.add_argument('--eta', type=float, default=0.1,
                        help='The parameter eta as in the paper. Defaults to 0.1')
    parser.add_argument('--steps_before_training', type=int, default=10000,
                        help='Steps before training. Defaults to 10,000.')
    parser.add_argument('--hyperopt', action='store_true',
                        help='Run hyperparameter optimisation.')
    parser.add_argument('--max_train_steps', type=int, default=2000000,
                        help='The maximum number of training steps to run.')
    args = parser.parse_args()

    if args.hyperopt:
        hypers_list = [sample_hypers() for i in range(100)]
    else:
        hypers = Hyperparameters(max_replay=200000, max_supervised=1000000,
                best_response_lr=1e-2, supervised_lr=5e-3,
                steps_before_training=args.steps_before_training, eta=args.eta,
                update_target_q_every=300, initial_epsilon=0.1, final_epsilon=0.0,
                epsilon_steps=10000, batch_size=128, q_learn_every=1,
                policy_learn_every=1, clip_reward=args.clip_reward,
                net_sizes=NetSizes(2, 64, 2, 64))
        hypers_list = [hypers]

    print("Using hyperparameters: {}".format(hypers_list))
    print("Training for {} steps".format(args.max_train_steps))

    for hypers in hypers_list:
        tf.reset_default_graph()
        print("Using hyperparameters: {}".format(hypers))
        cards = get_deck(num_values=args.num_values, num_suits=args.num_suits)
        nfsp(LeducNFSP(cards), hypers, max_train_steps=args.max_train_steps)
