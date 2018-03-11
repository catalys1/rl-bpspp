import itertools

import numpy as np

import gym
from probpy.inferencedriver import InferenceDriver


def _normalize(v, ord=1, axis=-1):
    norm = np.linalg.norm(v, ord=ord, axis=axis)
    if norm <= 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def explore(env, policy, render_mode=None):
    state = env.reset()
    total_reward = 0.0

    for step_count in itertools.count():
        action = np.random.choice(env.action_space.n, p=policy[state])
        obs, reward, done, _ = env.step(action)

        total_reward += reward

        if done:
            break
        state = obs

    # HACK so log works
    return np.clip(total_reward, np.finfo(dtype=float).eps, 1.)


def model(pp, env, actions):
    # HACK this should be a dirichlet
    bias = _normalize([
        pp.random(name='bias', loop_iter=i) for i in range(env.action_space.n)
    ])

    policy = np.zeros((env.observation_space.n, env.action_space.n),
                      dtype=np.int8)
    for i in range(env.observation_space.n):
        act = pp.choice(elements=actions, p=bias, name='policy', loop_iter=i)
        policy[i, act] = 1

    val = explore(env, policy)

    pp.choice(elements=[1, 0], p=[val, 1. - val], name='r')

    return val


if __name__ == '__main__':
    np.seterr(all='raise')
    num_samples = 100

    gym.envs.registration.register(
        id='NChain-custom-v0',
        entry_point='envs.modified_nchain:ModifiedNChainEnv',
        kwargs={
            'n': 5,
            'slip': 0.00,
            'intermediate': 0,
            'end': 1.
        },
        timestep_limit=100,
    )
    env = gym.make('NChain-custom-v0')

    # env = gym.wrappers.Monitor(env, 'logdir/')
    # [0, 1] == ['forward', 'backward']
    actions = np.arange(env.action_space.n, dtype=np.int8)

    driver = InferenceDriver(lambda pp: model(pp, env, actions))
    driver.condition(label='r-0', value=1)
    driver.init_model()
    # driver.prior(label="bias-0", value=.9)
    # driver.prior(label="bias-1", value=.1)

    driver.burn_in(steps=50)

    for k, v in driver.run_inference(interval=1, samples=num_samples).items():
        print(k, v)

    driver.plot_ll()

    rewards = driver.plot_model_results()
    print(len(rewards), rewards, rewards.count(1.), sep='\n')