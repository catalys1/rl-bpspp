from collections import OrderedDict
import itertools
import pprint
import time

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


def explore(env, policy, render_mode=None, render_step=1):
    state = env.reset()
    total_reward = 0.0
    for step_count in itertools.count():
        action = np.random.choice(env.action_space.n, p=policy[state])
        obs, reward, done, _ = env.step(action)

        if render_mode is not None and step_count % render_step == 0:
            time.sleep(.1)
            env.render(mode=render_mode)

        total_reward += reward

        if done:
            break
        state = obs

    return total_reward / step_count


if __name__ == '__main__':
    n_iter = 2000
    logdir = 'logdir'

    gym.envs.registration.register(
        id='NChain-custom-v0',
        entry_point='gym.envs.toy_text:NChainEnv',
        kwargs={'n': 5, 'slip': 0., 'small': 0.01, 'large': 1.},
        timestep_limit=1000,
    )
    env = gym.make('NChain-custom-v0')
    actions = ['F', 'B']

    # define the model
    bias = ed.models.Dirichlet(concentration=tf.zeros(env.action_space.n))
    bias_posterior = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, *bias.shape])))
    bias_proposal = ed.models.Normal(loc=bias, scale=.2)
    # policy = OrderedDict()
    # policy_posterior = OrderedDict()
    # latent_vars = OrderedDict({bias: bias_posterior})
    # latent_vars = [bias]
    # for i in range(env.observation_space.n):
    full_bias = tf.reshape(tf.tile(bias, [env.observation_space.n]), [env.observation_space.n, env.action_space.n])
    policy = ed.models.Multinomial(total_count=1., probs=full_bias)
    policy_posterier = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, *full_bias.shape])))
    policy_proposal = ed.models.Normal(loc=policy, scale=.2)

    policy_value = tf.placeholder(tf.float32)
    r = ed.models.Categorical(probs=[1. - policy_value, policy_value])

    inference = ed.MetropolisHastings(latent_vars={bias: bias_posterior, policy: policy_posterier},
                                      proposal_vars={bias: bias_proposal, policy: policy_proposal},
                                      data={r: 1.})
    inference.initialize()
    tf.global_variables_initializer().run()

    sess = ed.get_session()
    with tqdm.trange(inference.n_iter, desc='inference') as progress_bar:
        for _ in progress_bar:
            value = explore(env, sess.run(policy))
            info = inference.update(feed_dict={policy_value: value})
            progress_bar.set_postfix(accept_rate=info['accept_rate'], val=value)
        final_policy = [actions[np.argmax(a)] for a in sess.run(policy)]
        progress_bar.write(pprint.pformat(final_policy))
    inference.finalize()
