# coding: utf-8

import tensorflow as tf
import numpy as np
from leduc import PreFlopLeducHoldEm
from fullleduc import LeducHoldEm
import random
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

# Estimate the exploitability of the agent against a random strategy: how much
# does a random player win on average against the agent?
def estimate_exploitability_random(sess, game, agent, num_episodes=1000):
    random_rewards = []
    for episode in xrange(num_episodes):
        # Choose which of our two agents goes first
        first_player = np.random.choice([1,2])
        next_player, state, _, _ = game.reset(first_player)
        player = next_player
        terminal = False
        while not terminal:
            if player == 2:
                # Play randomly
                action = np.random.choice([0,1,2])
            else:
                # Use the average policy of player 1
                policy = agent.predict_policy(sess, [state])
                action = np.random.choice([0,1,2], p=policy.ravel())
            next_player, next_state, rewards, terminal = game.step(action)
            state = next_state
            player = next_player
        # rewards contains the rewards as a dictionary where 1 indexes the first
        # player and 2 is the key for the second player. We need the reward for
        # agent 2 (the best response player).
        random_rewards.append(rewards[2])
    return np.mean(random_rewards), np.std(random_rewards) / np.sqrt(num_episodes)

# Estimate the exploitability of agent 1 by agent 2. Thus, use the average
# policy of agent 1 and the best response (q-network) of agent 2. Since agent 2
# is playing an approximate best response strategy against agent 1, we treat
# this as a sample of the expected payoff of a best response against agent 1's
# average strategy (the policy network). The exploitability of a strategy is
# defined as the expected payoff of a best response strategy against it.
def estimate_exploitability(sess, game, agent1, agent2, strategy1='policy', strategy2='q', num_episodes=1000):
    agent2_rewards = []
    for episode in xrange(num_episodes):
        # Choose which of our two agents goes first
        first_player = np.random.choice([1,2])
        next_player, state, _, _ = game.reset(first_player)
        player = next_player
        terminal = False
        while not terminal:
            if player == 1:
                if strategy1 == 'q':
                    q_values = agent1.predict_q(sess, [state])
                    action = np.argmax(q_values, axis=1)
                else:
                    # Use the average policy of player 1
                    policy = agent1.predict_policy(sess, [state])
                    action = np.random.choice([0,1,2], p=policy.ravel())
            else:
                if strategy2 == 'q':
                    q_values = agent2.predict_q(sess, [state])
                    action = np.argmax(q_values, axis=1)
                else:
                    # Use the average policy of player 2
                    policy = agent2.predict_policy(sess, [state])
                    action = np.random.choice([0,1,2], p=policy.ravel())
            next_player, next_state, rewards, terminal = game.step(action)
            state = next_state
            player = next_player
        # rewards contains the rewards as a dictionary where 1 indexes the first
        # player and 2 is the key for the second player. We need the reward for
        # agent 2 (the best response player).
        agent2_rewards.append(rewards[2])
    return np.mean(agent2_rewards), np.std(agent2_rewards) / np.sqrt(num_episodes)

# Compute the best response to the agent's policy network strategy.
def best_response(sess, game, agent_to_eval, agent_br, num_train_steps=10000,
    steps_before_training=1000, update_target_q_every=300, q_learn_every=1,
    batch_size=128):
    agents = {1: agent_to_eval, 2: agent_br}
    q_losses = []
    for train_step in range(num_train_steps):
        # Play one game
        first_player = np.random.choice([1,2])
        transitions_br = []

        next_player, state, _, _ = game.reset(first_player)
        terminal = False
        player = next_player
        while not terminal:
            agent = agents[player]
            if player == 2:
                # Sample the action from the best response policy
                q_values = agent.predict_q(sess, [state])
                action = np.argmax(q_values, axis=1)[0]
            else:
                # Sample the action from the average policy
                policy = agent.predict_policy(sess, [state])
                action = np.random.choice([0,1,2], p=policy.ravel())

            # Take the step and get the new states
            next_player, next_state, rewards, terminal = game.step(action)

            # Add the transitions (ignoring reward and terminal for now)
            if player == 2:
                transitions_br.append({'state': state, 'next_state': next_state,
                'action': action, 'reward': 0.0, 'terminal': False})

            # Set the next player
            player = next_player
            state = next_state

        # The game just ended, so the last frame was terminal. The game returns
        # rewards in the order: first player, second player, so we assign them
        # to the correct agents.
        if len(transitions_br) > 0:
            transitions_br[-1]['reward'] = rewards[2]
            transitions_br[-1]['terminal'] = True
        agent_br.append_replay_memory(transitions_br)

        # Train the Q-network of the best response agent
        if train_step >= steps_before_training:
            if train_step % q_learn_every == 0:
                q_loss = agent_br.train_q_network(sess, batch_size)
                q_losses.append(q_loss)

            # Update the target networks
            if train_step % update_target_q_every == 0:
                print "Updating target network"
                agent_br.update_target_network(sess)

    return agent_br

def create_summary_tensors():
    # Create a dictionary of all summary nodes
    summary_tensor = {}
    summary_tensor['q_loss_1'] = tf.placeholder('float32', ())
    summary_tensor['q_loss_2'] = tf.placeholder('float32', ())
    summary_tensor['policy_loss_1'] = tf.placeholder('float32', ())
    summary_tensor['policy_loss_2'] = tf.placeholder('float32', ())
    summary_tensor['exploitability_1'] = tf.placeholder('float32', ())
    summary_tensor['exploitability_2'] = tf.placeholder('float32', ())
    summary_tensor['exploitability_random_1'] = tf.placeholder('float32', ())
    summary_tensor['exploitability_random_2'] = tf.placeholder('float32', ())
    summary_tensor['exploitability_of_policies'] = tf.placeholder('float32', ())
    tf.summary.scalar('q_loss_1', summary_tensor['q_loss_1'])
    tf.summary.scalar('q_loss_2', summary_tensor['q_loss_2'])
    tf.summary.scalar('policy_loss_1', summary_tensor['policy_loss_1'])
    tf.summary.scalar('policy_loss_2', summary_tensor['policy_loss_2'])
    tf.summary.scalar('exploitability_1', summary_tensor['exploitability_1'])
    tf.summary.scalar('exploitability_2', summary_tensor['exploitability_2'])
    tf.summary.scalar('exploitability_random_1', summary_tensor['exploitability_random_1'])
    tf.summary.scalar('exploitability_random_2', summary_tensor['exploitability_random_2'])
    tf.summary.scalar('exploitability_of_policies', summary_tensor['exploitability_of_policies'])
    return summary_tensor

# agents: a dictionary with keys 1, 2 and values the two agents.
def nfsp(game, update_target_q_every=1000, initial_epsilon=0.1, final_epsilon=0.0, epsilon_steps=100000, eta=0.1, max_train_steps=10000000, batch_size=128, steps_before_training=100000, q_learn_every=32, policy_learn_every=128, verbose=False, players_to_train=[1,2], clip_reward=True):

    # Create two agents
    agents = {1: Agent('1', game.state_dim()), 2: Agent('2', game.state_dim())}

    # Create summary tensors
    summary_tensor = create_summary_tensors()

    # Merge all summaries
    merged = tf.summary.merge_all()

    save_path = 'experiments/' + game.name() + '/' + strftime("%d-%m-%Y-%H:%M:%S", gmtime())

    if not os.path.exists(save_path):
        print "Path doesn't exist, so creating", save_path
        os.makedirs(save_path)

    # Create the session and initialise all variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf_train_writer = tf.summary.FileWriter(save_path, sess.graph)

    epsilon = initial_epsilon

    q_losses = {1: [], 2: []}
    policy_losses = {1: [], 2: []}

    # Update the target network to start with
    print "Updating target networks"
    for agent in agents.values():
        agent.update_target_network(sess)

    # Collect rollouts from a game with the two agents.
    for train_step in xrange(max_train_steps):
        # Choose random player to start the game
        first_player = np.random.choice([1,2])
        if verbose:
            print "First player", first_player

        transitions = {1: [], 2:[]}
        supervised = {1: [], 2: []}
        # Select the strategies
        strategy1 = np.random.choice(['q', 'policy'], p=[eta, 1.0-eta])
        strategy2 = np.random.choice(['q', 'policy'], p=[eta, 1.0-eta])
        strategies = {1: strategy1, 2: strategy2}

        if verbose:
            print "Strategies", strategies

        # Play one game
        next_player, state, _, _ = game.reset(first_player)
        terminal = False
        player = next_player
        while not terminal:
            if verbose:
                print "Player", player
                print "State", state
            agent = agents[player]
            strategy = strategies[player]
            # Sample the action from the corresponding policy
            if strategy == 'q':
                if verbose:
                    print "Playing with q"
                # Epsilon greedy strategy
                if np.random.random() < epsilon:
                    action = np.random.choice([0,1,2])
                else:
                    q_values = agent.predict_q(sess, [state])
                    action = np.argmax(q_values, axis=1)[0]
            else:
                if verbose:
                    print "Playing with policy"
                policy = agent.predict_policy(sess, [state])
                action = np.random.choice([0,1,2], p=policy.ravel())
            if verbose:
                print "Takes action", action
            next_player, next_state, rewards, terminal = game.step(action)
            if verbose:
                print "Next player", next_player
                print "Rewards", rewards

            # Add the transitions (ignoring reward and terminal for now, since we will update this later).
            transitions[player].append({'state': state, 'next_state': next_state, 'action': action, 'reward': 0.0, 'terminal': False})

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
                print "Adding transitions to player", player
                print transitions[player]
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
                        print "Updating target networks"
                    agent.update_target_network(sess)

        # Evaluate the best response network (the q network) for player 1 against
        # player 2's average policy (the policy network), and vice versa.
        if train_step % update_target_q_every == 0:
            print "Train step", train_step
            if train_step > steps_before_training:
                exploit1, exploit1_std = estimate_exploitability(sess, game, agents[1], agents[2])
                exploit2, exploit2_std = estimate_exploitability(sess, game, agents[2], agents[1])
                exploit_policies, exploit_policies_std = estimate_exploitability(sess, game, agents[1], agents[2], strategy1='policy', strategy2='policy')
                exploit1_random, exploit1_random_std = estimate_exploitability_random(sess, game, agents[1])
                exploit2_random, exploit2_random_std = estimate_exploitability_random(sess, game, agents[2])
                print "Exploitabilities: {} ({}), {} ({})".format(exploit1, exploit1_std, exploit2, exploit2_std)
                print "Exploitabilities (policy1 vs policy2): {} ({})".format(exploit_policies, exploit_policies_std)
                print "Exploitabilities against random: {} ({}), {} ({})".format(exploit1_random, exploit1_random_std, exploit2_random, exploit2_random_std)
                print "Q losses:", np.mean(q_losses[1]), np.mean(q_losses[2])
                print "Policy losses:", np.mean(policy_losses[1]), np.mean(policy_losses[2])
                print "Epsilon", epsilon
                print "-------------------"

                summary = sess.run(merged, feed_dict={
                    summary_tensor['q_loss_1']: np.mean(q_losses[1]),
                    summary_tensor['q_loss_2']: np.mean(q_losses[2]),
                    summary_tensor['policy_loss_1']: np.mean(policy_losses[1]),
                    summary_tensor['policy_loss_2']: np.mean(policy_losses[2]),
                    summary_tensor['exploitability_1']: exploit1,
                    summary_tensor['exploitability_2']: exploit2,
                    summary_tensor['exploitability_random_1']: exploit1_random,
                    summary_tensor['exploitability_random_2']: exploit2_random,
                    summary_tensor['exploitability_of_policies']: exploit_policies
                })
                tf_train_writer.add_summary(summary, train_step)

            #best_response(sess, game, agents[1], agent_br)
            #print "Computed best response"
            #print estimate_exploitability(sess, game, agents[1], agent_br)

    return agents

if __name__ == '__main__':
    nfsp(LeducHoldEm(), verbose=False, eta=0.2)
