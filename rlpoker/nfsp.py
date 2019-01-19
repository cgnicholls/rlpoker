# coding: utf-8

import tensorflow as tf
import numpy as np
from games.leduc import Leduc
from rlpoker.best_response import compute_exploitability
from agent import Agent
from time import time, gmtime, strftime
import os

def compute_epsilon(initial_epsilon, final_epsilon, train_step, epsilon_steps):
    train_fraction = float(train_step) / float(epsilon_steps)
    if train_fraction < 0.0:
        train_fraction = 0.0
    if train_fraction > 1.0:
        train_fraction = 1.0
    return (1-train_fraction) * initial_epsilon + train_fraction * final_epsilon


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
def nfsp(game, update_target_q_every=1000, initial_epsilon=0.1, final_epsilon=0.0, epsilon_steps=100000, eta=0.1, max_train_steps=10000000, batch_size=128, steps_before_training=100000, q_learn_every=32, policy_learn_every=128, verbose=False, players_to_train=[1,2], clip_reward=True):

    # Create two agents
    agents = {1: Agent('1', game.state_dim()), 2: Agent('2', game.state_dim())}

    # Create summary tensors
    summary_tensor = create_summary_tensors()

    # Merge all summaries
    merged = tf.summary.merge_all()

    time_str = strftime("%d-%m-%Y-%H:%M:%S", gmtime())
    save_path = os.path.join('experiments', game.name(), time_str)

    if not os.path.exists(save_path):
        print("Path doesn't exist, so creating: {}".format(save_path))
        os.makedirs(save_path)

    # Create the session and initialise all variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf_train_writer = tf.summary.FileWriter(save_path, sess.graph)

    epsilon = initial_epsilon

    q_losses = {1: [], 2: []}
    policy_losses = {1: [], 2: []}

    # Update the target network to start with
    print("Updating target networks")
    for agent in agents.values():
        agent.update_target_network(sess)

    # Collect rollouts from a game with the two agents.
    for train_step in range(max_train_steps):
        # Choose random player to start the game
        first_player = np.random.choice([1,2])
        if verbose:
            print("First player: {}".format(first_player))

        transitions = {1: [], 2:[]}
        supervised = {1: [], 2: []}
        # Select the strategies
        strategy1 = np.random.choice(['q', 'policy'], p=[eta, 1.0-eta])
        strategy2 = np.random.choice(['q', 'policy'], p=[eta, 1.0-eta])
        strategies = {1: strategy1, 2: strategy2}

        if verbose:
            print("Strategies: {}".format(strategies))

        # Play one game
        next_player, state, _, _ = game.reset(first_player)
        terminal = False
        player = next_player
        while not terminal:
            if verbose:
                print("Player: {}".format(player))
                print("State: {}".format(state))
            agent = agents[player]
            strategy = strategies[player]
            # Sample the action from the corresponding policy
            if strategy == 'q':
                if verbose:
                    print("Playing with q")
                # Epsilon greedy strategy
                if np.random.random() < epsilon:
                    action = np.random.choice([0,1,2])
                else:
                    q_values = agent.predict_q(sess, [state])
                    action = np.argmax(q_values, axis=1)[0]
            else:
                if verbose:
                    print("Playing with policy")
                policy = agent.predict_policy(sess, [state])
                action = np.random.choice([0,1,2], p=policy.ravel())
            if verbose:
                print("Takes action: {}".format(action))
            next_player, next_state, rewards, terminal = game.step(action)
            if verbose:
                print("Next player: {}".format(next_player))
                print("Rewards: {}".format(rewards))

            # Add the transitions (ignoring reward and terminal for now,
            # since we will update this later).
            transitions[player].append({'state': state,
                                        'next_state': next_state,
                                        'action': action,
                                        'reward': 0.0,
                                        'terminal': False})

            # Only add to the supervised learning memory if we were playing our
            # best response strategy.
            if strategy == 'q':
                supervised[player].append({'state': state, 'action': action})

            # Set the next player and next state
            player = next_player
            state = next_state

        # The game just ended, so the last frame was terminal. The game returns
        # rewards in the order: first player, second player, so we assign them
        # to the correct agents.

        for player in [1,2]:
            if len(transitions[player]) > 0:
                if clip_reward:
                    reward = np.clip(rewards[player], -1.0, 1.0)
                else:
                    reward = rewards[player]
                transitions[player][-1]['reward'] = reward
                transitions[player][-1]['terminal'] = True
            if verbose:
                print("Adding transitions to player: {}".format(player))
                print(transitions[player])
            agents[player].append_replay_memory(transitions[player])
            agents[player].append_supervised_memory(supervised[player])

        # Train the Q-networks
        if train_step >= steps_before_training:
            epsilon = compute_epsilon(initial_epsilon, final_epsilon, train_step - steps_before_training, epsilon_steps)
            for player in [1,2]:
                if not player in players_to_train:
                    continue
                agent = agents[player]
                if train_step % q_learn_every == 0:
                    for i in range(2):
                        q_loss = agent.train_q_network(sess, batch_size)
                        q_losses[player].append(q_loss)
                if train_step % policy_learn_every == 0:
                    for i in range(2):
                        policy_loss = agent.train_policy_network(sess, batch_size)
                        policy_losses[player].append(policy_loss)

                # Update the target networks
                if train_step % update_target_q_every == 0:
                    if player == 1:
                        print("Updating target networks")
                    agent.update_target_network(sess)

        # Evaluate the best response network (the q network) for player 1 against
        # player 2's average policy (the policy network), and vice versa.
        if train_step % update_target_q_every == 0:
            print("Train step: {}".format(train_step))
            if train_step > steps_before_training:
                exploit1 = agents[1].compute_exploitability(game)
                exploit2 = agents[2].compute_exploitability(game)
                print("Exploitabilities: {}, {}".format(exploit1, exploit2))
                print("Q losses: {}, {}".format(np.mean(q_losses[1]),
                                                np.mean(q_losses[2])))
                print("Policy losses: {}, {}".format(
                    np.mean(policy_losses[1]),
                    np.mean(policy_losses[2])))
                print("Epsilon: {}".format(epsilon))
                print("-------------------")

                summary = sess.run(merged, feed_dict={
                    summary_tensor['q_loss_1']: np.mean(q_losses[1]),
                    summary_tensor['q_loss_2']: np.mean(q_losses[2]),
                    summary_tensor['policy_loss_1']: np.mean(policy_losses[1]),
                    summary_tensor['policy_loss_2']: np.mean(policy_losses[2]),
                    summary_tensor['exploitability_1']: exploit1,
                    summary_tensor['exploitability_2']: exploit2
                })
                tf_train_writer.add_summary(summary, train_step)

    return agents

if __name__ == '__main__':
    nfsp(Leduc(), verbose=False, eta=0.2)
