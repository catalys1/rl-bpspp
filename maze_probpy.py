import collections
import itertools

import gym
import gym_openmaze
import numpy as np

from probpy.inferencedriver import InferenceDriver


def _normalize(v, ord=1, axis=-1):
    norm = np.linalg.norm(v, ord=ord, axis=axis)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def explore(env, policy, render_mode=None, max_previous_states=4):
    total_reward = 0.0
    state = tuple(env.reset())

    previous_states = collections.deque(maxlen=max_previous_states)
    for i in itertools.count():
        previous_states.append(state)
        action = np.random.choice(env.action_space.n, p=policy[state])
        observation, reward, done, info = env.step(action)

        if render_mode is not None:
            env.render(mode=render_mode)

        total_reward += reward

        obs = tuple(observation)
        if (
            len(previous_states) >= max_previous_states
            and obs in previous_states
        ) or done:
            break
        state = obs

    # HACK so log works
    return np.clip(total_reward / i, np.finfo(dtype=float).eps, 1.)


def model(pp, env, all_actions):
    # HACK this should be a dirichlet
    bias = _normalize([
        pp.random(name='bias', loop_iter=i) for i in range(env.action_space.n)
    ])

    policy = np.zeros((*env.observation_space.shape, env.action_space.n),
                      dtype=np.int8)
    it = 0
    for y in range(policy.shape[0]):
        for x in range(policy.shape[1]):
            available_actions = env.action_space.available_actions(y, x)
            if not np.any(available_actions):
                continue
            local_bias = _normalize(np.multiply(bias, available_actions))
            index = pp.choice(
                elements=all_actions, p=local_bias, name='policy', loop_iter=it
            )
            policy[y, x, index] = 1  # one-hot, eg: [0, 1, 0, 0] == 'E'
            it += 1

    val = explore(env, policy)

    pp.choice(elements=[1., 0.], p=[val, 1. - val], name='r')

    return val


if __name__ == '__main__':
    np.seterr(all='raise')

    env = gym.make('openmazeurgency-v0')
    # [0, 1, 2, 3] == ['N', 'E', 'S', 'W']
    all_actions = np.arange(env.action_space.n)

    driver = InferenceDriver(lambda pp: model(pp, env, all_actions))
    driver.condition(label='r-0', value=1.)
    driver.init_model()
    driver.prior(label="bias-0", value=.575)
    driver.prior(label="bias-1", value=.2)
    driver.prior(label="bias-2", value=.025)
    driver.prior(label="bias-3", value=.2)

    driver.burn_in(steps=1000)

    for k, v in driver.run_inference(interval=1, samples=5000).items():
        print(k, v)

    driver.plot_ll()

    rewards = driver.plot_model_results()
    print(len(rewards), rewards, rewards.count(1.), sep='\n')
