import itertools
import collections

import edward as ed
import gym
import numpy as np
import tensorflow as tf
import tqdm

# tf.contrib.eager.enable_eager_execution()


def run_maze(env, policy, render_mode=None, render_step=1, max_previous_states=2):
    total_reward = 0.0
    state = tuple(env.reset())
    # path = {state}
    previous_states = collections.deque(maxlen=max_previous_states)
    for i in itertools.count():
        previous_states.append(state)
        action = np.random.choice(env.action_space.n, p=policy[state])
        observation, reward, done, info = env.step(action)

        if render_mode is not None and i % render_step == 0:
            env.render(mode=render_mode)

        total_reward += reward

        obs = tuple(observation)
        # print(state, obs, state == obs)
        if (len(previous_states) >= max_previous_states and obs in previous_states) or done:
            break
        state = obs
        # path.add(state)
    return total_reward, i


def propose_policy(current_policy):
    print(current_policy[1, 1, 1])
    return current_policy


if __name__ == '__main__':
    n_iter = 100000
    n_samples = 10
    logdir = 'logdir'

    gym.envs.registration.register(
        id='openmaze-custom-v0',
        entry_point='gym_openmaze.envs.openmaze:OpenMaze',
        kwargs={'size': (10, 7), 'random': False},
        max_episode_steps=100000
    )
    env = gym.make('openmaze-custom-v0')  # TODO: change to random
    policy_shape = [*env.observation_space.shape, env.action_space.n]

    with tf.name_scope("model"):
        bias = np.array([.575, .2, .025, .2], dtype=np.float32)
        policy = ed.models.Multinomial(total_count=1., probs=np.broadcast_to(bias, policy_shape))
        # policy = ed.models.DirichletMultinomial(total_count=1., concentration=tf.ones(policy_shape))
        proposal_policy = ed.models.Normal(loc=policy, scale=10., value=policy)

        policy_value = tf.placeholder(tf.float32)
        r = ed.models.Binomial(total_count=1., probs=policy_value, value=1.)
        # policy_value = ed.models.Normal(loc=.5, scale=0.5)
        # policy_value_proposal = ed.models.Normal(loc=policy_value, scale=0.5)
        # policy_value_proposal = tf.placeholder(tf.float32)

    with tf.name_scope("posterior"):
        q_policy = ed.models.Empirical(params=tf.Variable(tf.zeros([n_iter, *policy_shape])))

    inference = ed.MetropolisHastings(latent_vars={policy: q_policy},
                                      proposal_vars={policy: proposal_policy},
                                      data={r: 1.0})

    inference.initialize(logdir=logdir)
    tf.global_variables_initializer().run()

    with tqdm.trange(inference.n_iter, desc='inference') as progress_bar:
        for _ in progress_bar:
            v, i = run_maze(env, ed.get_session().run(policy))
            info = inference.update(feed_dict={policy_value: v})
            progress_bar.set_postfix({'accept_rate': info['accept_rate'], 'val': v, 'maze_it': i})
    run_maze(env, ed.get_session().run(policy), render_mode='human')
    inference.finalize()
