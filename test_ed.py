import collections
import itertools
import pprint

import edward as ed
import gym
import gym_openmaze
import numpy as np
import tensorflow as tf
import tqdm


def _normalize(v, ord=1, axis=-1):
    norm = np.linalg.norm(v, ord=ord, axis=axis)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def run_maze(env, policy, render_mode=None, render_step=1, max_previous_states=4, max_duplicate_states=None):
    if max_duplicate_states is None:
        max_duplicate_states = max(1, round(max_previous_states / 3))

    previous_states = collections.deque(maxlen=max_previous_states)
    state = tuple(env.reset())
    total_reward = 0.0
    for step_count in itertools.count():
        previous_states.append(state)
        action = np.random.choice(env.action_space.n, p=policy[state])
        observation, reward, done, _ = env.step(action)

        if render_mode is not None and step_count % render_step == 0:
            env.render(mode=render_mode)

        total_reward += reward

        obs = tuple(observation)
        if (len(previous_states) >= max_previous_states and previous_states.count(obs) >= max_duplicate_states) or done:
            break
        state = obs

    return total_reward, step_count


if __name__ == '__main__':
    n_iter = 1000
    logdir = 'logdir'

    env = gym.make('CartPole-v0')

    actions = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    policy_shape = [*env.observation_space.shape, 1]
    policy_len = np.multiply(*env.observation_space.shape)


    sess = ed.get_session()

    # define the model
    # bias = np.array([.575, .2, .025, .2], dtype=np.float32)  # TODO: learn this
    bias = ed.models.Dirichlet(concentration=tf.ones(env.action_space.n), value=[.575, .2, .025, .2])
    bias_posterior = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, *bias.shape])))
    bias_proposal = ed.models.Normal(loc=bias, scale=.9)
    policy = {}
    latent_vars = {bias: bias_posterior}
    proposal_vars = {bias: bias_proposal}
    it = 0
    for y in range(policy_shape[0]):
        for x in range(policy_shape[1]):
            avail_actions = env.action_space.available_actions(y, x)

            print(avail_actions)

            b = tf.norm(tf.multiply(bias, avail_actions), ord=1)
            pi = ed.models.Multinomial(total_count=1., probs=b)
            pi_posterier = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, *b.shape])))
            pi_proposal = ed.models.Normal(loc=pi, scale=.9, value=pi)

            policy[(y, x)] = pi
            latent_vars[pi] = pi_posterier
            proposal_vars[pi] = pi_proposal
            it += 1
    policy_value = tf.placeholder(tf.float32)
    r = ed.models.Categorical(probs=[1. - policy_value, policy_value])

    inference = ed.MetropolisHastings(latent_vars=latent_vars, proposal_vars=proposal_vars, data={r: 1})
    inference.initialize(logdir=logdir)
    tf.global_variables_initializer().run()

    with tqdm.trange(inference.n_iter, desc='inference') as progress_bar:
        for _ in progress_bar:
            value, i = run_maze(env, {key: sess.run(val) for key, val in policy.items()})
            info = inference.update(feed_dict={policy_value: value})
            progress_bar.set_postfix(accept_rate=info['accept_rate'], val=value, maze_it=i)
        final_policy = {key: actions[int(np.argmax(sess.run(val)))] for key, val in policy.items()}
        progress_bar.write(pprint.pformat(final_policy))
    inference.finalize()
