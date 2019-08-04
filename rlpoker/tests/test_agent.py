import numpy as np
import tensorflow as tf

from rlpoker.agent import Agent


def test_update_target_network():
    input_dim = 10
    action_dim = 3
    agent = Agent('A', input_dim, action_dim)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # First run q network.
        state = np.random.randn(4, input_dim)
        predict_q_func = agent.predict_q(sess, state)
        predict_q = sess.run(agent.q_network['output'], feed_dict={
            agent.q_network['input']: state,
            agent.q_network['training']: False
        })
        np.testing.assert_allclose(predict_q, predict_q_func)

        # Then predict with target network.
        predict_target_q = sess.run(agent.target_q_network['output'], feed_dict={
            agent.target_q_network['input']: state,
            agent.target_q_network['training']: False
        })
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(predict_q, predict_target_q)

        # Now update target network and check it agrees with the q network.
        agent.update_target_network(sess)

        predict_target_q = sess.run(agent.target_q_network['output'], feed_dict={
            agent.target_q_network['input']: state,
            agent.target_q_network['training']: False
        })
        np.testing.assert_allclose(predict_q, predict_target_q)

