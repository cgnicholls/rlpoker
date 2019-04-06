# coding: utf-8

import argparse
from time import gmtime, strftime
import os

import numpy as np
import tensorflow as tf

from rlpoker.nfsp_game import NFSPGame
from rlpoker.games.leduc import LeducNFSP
from rlpoker.games.card import get_deck
from rlpoker.agent import Agent
from rlpoker.best_response import compute_exploitability


def compute_epsilon(initial_epsilon, final_epsilon, train_step, epsilon_steps):
    train_fraction = float(train_step) / float(epsilon_steps)
    if train_fraction < 0.0:
        train_fraction = 0.0
    if train_fraction > 1.0:
        train_fraction = 1.0
    return (1-train_fraction) * initial_epsilon + train_fraction * final_epsilon


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


def create_summary_tensors():
    # Create a dictionary of all summary nodes
    summary_tensor = dict()
    summary_tensor['q_loss_1'] = tf.placeholder('float32', ())
    summary_tensor['q_loss_2'] = tf.placeholder('float32', ())
    summary_tensor['policy_loss_1'] = tf.placeholder('float32', ())
    summary_tensor['policy_loss_2'] = tf.placeholder('float32', ())
    summary_tensor['exploitability_1'] = tf.placeholder('float32', ())
    summary_tensor['exploitability_2'] = tf.placeholder('float32', ())
    tf.summary.scalar('q_loss_1', summary_tensor['q_loss_1'])
    tf.summary.scalar('q_loss_2', summary_tensor['q_loss_2'])
    tf.summary.scalar('policy_loss_1', summary_tensor['policy_loss_1'])
    tf.summary.scalar('policy_loss_2', summary_tensor['policy_loss_2'])
    tf.summary.scalar('exploitability_1', summary_tensor['exploitability_1'])
    tf.summary.scalar('exploitability_2', summary_tensor['exploitability_2'])
    return summary_tensor


# agents: a dictionary with keys 1, 2 and values the two agents.
def nfsp(game, update_target_q_every=300, initial_epsilon=0.1, final_epsilon=0.0, epsilon_steps=100000, eta=0.1,
         max_train_steps=10000000, batch_size=128, steps_before_training=10000, q_learn_every=1,
         policy_learn_every=1, clip_reward=True, best_response_lr=1e-1, supervised_lr=5e-3, train_players=(1, 2)):

    # Create two agents
    agents = {1: Agent('1', game.state_dim, game.action_dim,
                       best_response_lr=best_response_lr, supervised_lr=supervised_lr),
              2: Agent('2', game.state_dim, game.action_dim,
                       best_response_lr=best_response_lr, supervised_lr=supervised_lr)}

    # Create summary tensors
    summary_tensor = create_summary_tensors()

    # Merge all summaries
    merged = tf.summary.merge_all()

    time_str = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    save_path = os.path.join('experiments', time_str)

    if not os.path.exists(save_path):
        print("Path doesn't exist, so creating: {}".format(save_path))
        os.makedirs(save_path)

    log_file = os.path.join(save_path, 'nfsp.log')

    print("Log file {}".format(log_file))

    # Create the session and initialise all variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf_train_writer = tf.summary.FileWriter(save_path, sess.graph)

    epsilon = initial_epsilon

    q_losses = {1: [], 2: []}
    policy_losses = {1: [], 2: []}

    # Update the target network to start with
    with open(log_file, 'a') as f:
        f.write("Updating target networks\n")

    for agent in agents.values():
        agent.update_target_network(sess)

    # Collect rollouts from a game with the two agents.
    for train_step in range(max_train_steps):
        # Choose random player to start the game
        first_player = np.random.choice([1, 2])
        # first_player = 1
        with open(log_file, 'a') as f:
            f.write("First player: {}\n".format(first_player))

        states = {1: [], 2: []}
        actions = {1: [], 2: []}
        supervised = {1: [], 2: []}
        # Select the strategies
        strategy1 = np.random.choice(['q', 'policy'], p=[eta, 1.0-eta])
        strategy2 = np.random.choice(['q', 'policy'], p=[eta, 1.0-eta])
        strategies = {1: strategy1, 2: strategy2}

        with open(log_file, 'a') as f:
            f.write("Strategies: {}\n".format(strategies))

        # Play one game
        next_player, state, available_actions, _, _ = game.reset(first_player)
        with open(log_file, 'a') as f:
            f.write("Current node: {}\n".format(game._current_node))
        terminal = False
        player = next_player
        while not terminal:
            with open(log_file, 'a') as f:
                f.write("Player: {}\n".format(player))
                f.write("State: {}\n".format(state))
            agent = agents[player]
            strategy = strategies[player]
            # Sample the action from the corresponding policy
            if strategy == 'q':
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
                with open(log_file, 'a') as f:
                    f.write("Playing with policy\n")
                policy = agent.predict_policy(sess, np.array([state])).ravel()
                policy = normalise_policy(policy, available_actions)

                # We first normalise the probabilities to the available
                # actions.
                action = np.random.choice([0, 1, 2], p=policy)

            with open(log_file, 'a') as f:
                f.write("Takes action: {}\n".format(action))

            next_player, next_state, available_actions, rewards, terminal = game.step(action)

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
        if clip_reward:
            rewards = {k: np.clip(v, -1.0, 1.0) for k, v in rewards.items()}

        transitions = build_transitions(states, actions, rewards)

        with open(log_file, 'a') as f:
            f.write("Terminal node: {}\n".format(game._current_node))

        # The game just ended, so the last frame was terminal. The game returns
        # rewards in the order: first player, second player, so we assign them
        # to the correct agents.

        for player in [1, 2]:
            with open(log_file, 'a') as f:
                f.write("Adding transitions to player: {}\n".format(player))
                f.write(str(transitions[player]) + '\n')
            agents[player].append_replay_memory(transitions[player])
            agents[player].append_supervised_memory(supervised[player])

        # Train the Q-networks
        if train_step >= steps_before_training:
            epsilon = compute_epsilon(initial_epsilon, final_epsilon,
                                      train_step - steps_before_training,
                                      epsilon_steps)
            for player, agent in agents.items():
                if player not in train_players:
                    continue
                if train_step % q_learn_every == 0:
                    q_loss = agent.train_q_network(sess, batch_size)
                    q_losses[player].append(q_loss)
                if train_step % policy_learn_every == 0:
                    policy_loss = agent.train_policy_network(sess, batch_size)
                    policy_losses[player].append(policy_loss)

                # Update the target networks
                if train_step % update_target_q_every == 0:
                    with open(log_file, 'a') as f:
                        f.write("Updating target network of {}\n".format(player))
                    agent.update_target_network(sess)

        # Evaluate the best response network (the q network) for player 1
        # against player 2's average policy (the policy network), and vice
        # versa.
        if train_step % update_target_q_every == 0:
            with open(log_file, 'a') as f:
                f.write("Train step: {}\n".format(train_step))
            if train_step > steps_before_training:
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

                summary = sess.run(merged, feed_dict={
                    summary_tensor['q_loss_1']: np.mean(q_losses[1]),
                    summary_tensor['q_loss_2']: np.mean(q_losses[2]),
                    summary_tensor['policy_loss_1']: np.mean(policy_losses[1]),
                    summary_tensor['policy_loss_2']: np.mean(policy_losses[2]),
                    summary_tensor['exploitability_1']: exploit1,
                    summary_tensor['exploitability_2']: exploit2
                })
                tf_train_writer.add_summary(summary, train_step)

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
    args = parser.parse_args()

    cards = get_deck(num_values=args.num_values, num_suits=args.num_suits)
    nfsp(LeducNFSP(cards), eta=args.eta, clip_reward=args.clip_reward,
         steps_before_training=args.steps_before_training)
