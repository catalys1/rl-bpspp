import itertools
import pprint

import gym
import gym_openmaze
import numpy as np

from inferencedriver import InferenceDriver


def _normalize(v, ord=1, axis=-1):
    norm = np.linalg.norm(v, ord=ord, axis=axis)
    if norm <= 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def explore(env, policy, render_mode=None, progress_bar=None):
    if render_mode == 'human':
        progress_bar.write('\n{}'.format(policy))

    state = env.reset()
    total_reward = 0.0

    for step_count in itertools.count():
        action = np.random.choice(env.action_space.n, p=policy[state])
        obs, reward, done, _ = env.step(action)

        total_reward += reward

        if done:
            break
        state = obs

    return total_reward / step_count, step_count


def model(pp, env, current_step, total_steps=0, progress_bar=None):
    # HACK
    bias = np.zeros(env.action_space.n)
    for i in range(env.action_space.n):
        bias[i] = pp.random(name='bias', loop_iter=i)
    bias = _normalize(bias)

    actions = np.arange(env.action_space.n, dtype=np.int8)
    policy = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.int8)
    for i in range(env.observation_space.n):
        action_idx = pp.choice(elements=actions, p=bias, name='policy', loop_iter=i)
        policy[i, action_idx] = 1
    # m = 'human' if total_steps == current_step + 1 else None
    val, _ = explore(env, policy, render_mode=None, progress_bar=progress_bar)

    pp.choice(elements=[1, 0], p=[min(1, val), max(0, 1. - val)], name='r')


if __name__ == '__main__':
    np.seterr(all='raise')
    num_samples = 1000

    gym.envs.registration.register(
        id='NChain-custom-v0',
        entry_point='gym.envs.toy_text:NChainEnv',
        kwargs={'n': 10, 'slip': 0.01, 'small': 0.01, 'large': 1.},
        timestep_limit=500,
    )
    env = gym.make('NChain-custom-v0')
    actions = ['F', 'B']

    driver = InferenceDriver(lambda pp, i=0, bar=None: model(pp, env, i, total_steps=num_samples, progress_bar=bar))

    driver.condition(label='r-0', value=1)

    driver.init_model()
    # driver.prior(label="bias-0", value=.9)
    # driver.prior(label="bias-1", value=.1)

    driver.burn_in(steps=500)

    for k, v in sorted(driver.run_inference(interval=2, samples=num_samples).items()):
        print(k, v)
