import itertools
import sys
import time

import gym
import numpy as np

from probpy.inferencedriver import InferenceDriver


def _normalize(v, ord=1, axis=-1):
    norm = np.linalg.norm(v, ord=ord, axis=axis)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def explore(env, policy, render_mode=None):
    total_reward = 0.0
    state = env.reset()

    for i in itertools.count():
        action = np.random.choice(env.action_space.n, p=policy[state])
        state, reward, done, info = env.step(action)

        if render_mode is not None:
            env.render(mode=render_mode)
            time.sleep(.1)

        total_reward += reward

        if done:
            break

    # HACK: so log works
    small = np.finfo(dtype=float).eps
    large = env.reward_range[1]
    return np.clip(total_reward, 0., large)


def model(pp, env, actions, render_mode=None):
    # HACK: this should be a dirichlet
    global_bias = _normalize([
        pp.random(name='bias', loop_iter=i) for i in range(env.action_space.n)
    ])

    policy_shape = (*env.observation_space.shape, env.action_space.n)
    policy = np.zeros(policy_shape, dtype=np.int8)
    i = 0
    for y in range(policy.shape[0]):
        for x in range(policy.shape[1]):
            available_actions = env.action_space.available_actions(y, x)
            if np.any(available_actions):
                bias = _normalize(np.multiply(global_bias, available_actions))
                name = 'policy_{}_{}_'.format(y, x)
                action_idx = pp.choice(name, actions, p=bias, loop_iter=i)
                # one-hot, eg: [0, 1, 0, 0] == 'E'
                policy[y, x, action_idx] = 1
            i += 1

    val = explore(env, policy, render_mode=render_mode)

    pp.choice(elements=[1, 0], p=[val, env.reward_range[1] - val], name='r')

    return val


if __name__ == '__main__':
    import gym_openmaze

    np.seterr(all='warn')

    # env_id = 'OpenMazeUrgencyMax10CyclesInWindowSize2-v0'
    # env_id = 'OpenMazeUrgencyMax4CyclesInWindowSize2-v0'
    env_id = 'OpenMazeOnlyCompletionRewardMax4CyclesInWindowSize2-v0'
    env = gym.make(env_id)

    try:
        policy = np.load(sys.argv[1])
        explore(env, policy, render_mode='human')
        sys.exit()
    except IndexError:
        pass

    # env = gym.wrappers.Monitor(env, 'logdir/{}'.format(env_id), force=True)
    # [0, 1, 2, 3] == ['N', 'E', 'S', 'W']
    actions = range(env.action_space.n)

    driver = InferenceDriver(lambda pp: model(pp, env, actions))
    driver.condition(label='r-0', value=1)
    driver.init_model()
    # driver.prior(label="bias-0", value=.575)
    # driver.prior(label="bias-1", value=.2)
    # driver.prior(label="bias-2", value=.025)
    # driver.prior(label="bias-3", value=.2)

    driver.burn_in(steps=100)

    policy_shape = (*env.observation_space.shape, env.action_space.n)
    posterior_policy = np.zeros(policy_shape, dtype=np.int8)
    for k, v in driver.run_inference(interval=1, samples=500).items():
        print(k, {
            'value': v['value'],
            'probs': v['parameters'].get('p'),
            'likelihood': v['likelihood']
        })
        if 'policy' in k:
            _, y, x, _ = k.split('_')
            posterior_policy[int(y), int(x), int(v['value'])] = 1

    # driver.plot_ll()

    # rewards = driver.plot_model_results()
    # print('avg:', np.mean(rewards))
    # print('completion count:',
    #       np.count_nonzero(np.isclose(rewards, env.reward_range[1])))

    explore(env, posterior_policy, render_mode='human')
