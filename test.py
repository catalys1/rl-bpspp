import collections
import itertools

import gym
import numpy as np

from inferencedriver import InferenceDriver


def _normalize(v, ord=1, axis=-1):
    norm = np.linalg.norm(v, ord=ord, axis=axis)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def run_maze(env, policy, render_mode=None, render_step=1, max_previous_states=4):
    total_reward = 0.0
    state = tuple(env.reset())

    previous_states = collections.deque(maxlen=max_previous_states)
    for i in itertools.count():
        previous_states.append(state)
        # action = np.random.choice(env.action_space.n, p=policy[state])
        action = policy[state]
        observation, reward, done, info = env.step(action)

        if render_mode is not None and i % render_step == 0:
            env.render(mode=render_mode)

        total_reward += reward

        obs = tuple(observation)
        if (len(previous_states) >= max_previous_states and obs in previous_states) or done:
            break
        state = obs

    return total_reward, i


def model(pp, env, current_step, total_steps):
    # HACK
    bias = np.zeros(env.action_space.n)
    for i in range(env.action_space.n):
        bias[i] = pp.random(name='bias', loop_iter=i)
    bias = _normalize(bias)

    policy = np.zeros((*env.observation_space.shape, 1), dtype=np.int8)
    it = 0
    for idx, _ in np.ndenumerate(policy):
        # [0, 1, 2, 3] == ['N', 'E', 'S', 'W']
        available_actions = env.action_space.available_actions(idx[0], idx[1])
        if not available_actions:
            continue
        local_bias = _normalize([bias[a] for a in available_actions])  # TODO: assumes available_actions is sorted
        policy[idx] = pp.choice(elements=available_actions, p=local_bias, name='policy', loop_iter=it)
        it += 1
    val, _ = run_maze(env, policy, render_mode='human' if total_steps - current_step < 10 else None)

    pp.choice(elements=[1., 0.], p=[val, 1. - val], name='r')


if __name__ == '__main__':
    num_samples = 5000

    gym.envs.registration.register(
        id='openmaze-custom-v0',
        entry_point='gym_openmaze.envs.openmaze:OpenMaze',
        kwargs={'size': (10, 7), 'random': False},
        max_episode_steps=100000
    )
    env = gym.make('openmaze-custom-v0')  # TODO: change to random

    driver = InferenceDriver(lambda pp, i: model(pp, env, current_step=i, total_steps=num_samples))

    driver.condition(label='r-0', value=1.)

    driver.init_model()
    driver.prior(label="bias-0", value=.575)
    driver.prior(label="bias-1", value=.2)
    driver.prior(label="bias-2", value=.025)
    driver.prior(label="bias-3", value=.2)

    driver.burn_in(steps=1000)

    driver.run_inference(interval=5, samples=num_samples)
