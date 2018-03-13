import itertools
import multiprocessing
import pickle
import sys
import time

import gym
import gym_openmaze
import numpy as np

from probpy.inferencedriver import InferenceDriver

np.seterr(all='warn')


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
            time.sleep(.5)

        total_reward += reward

        if done:
            break

    # HACK: so log works
    small = np.finfo(dtype=float).eps
    large = env.reward_range[1]
    return np.clip(total_reward, small, large)


def model(pp, env, actions, global_bias=None, render_mode=None):
    if global_bias is None:
        # learned bias
        # HACK: this should be a dirichlet
        global_bias = _normalize([
            pp.random(name='bias', loop_iter=i)
            for i in range(env.action_space.n)
        ])
    policy_shape = (*env.observation_space.shape, env.action_space.n)
    policy = np.zeros(policy_shape, dtype=np.float64)
    i = 0
    for y in range(policy.shape[0]):
        for x in range(policy.shape[1]):
            available_actions = env.action_space.available_actions(y, x)
            if np.any(available_actions):
                bias = _normalize(np.multiply(global_bias, available_actions))
                action_sum = np.zeros(env.action_space.n)
                for j in range(11):
                    name = 'policy{}_{}_{}_'.format(j, y, x)
                    action = np.zeros(env.action_space.n)
                    action_idx = pp.choice(name, actions, p=bias, loop_iter=i)
                    action[action_idx] = 1
                    # print('before', action_sum)
                    action_sum += action
                    # print('after', action_sum)
                # one-hot, eg: [0, 1, 0, 0] == 'E'
                policy[y, x] = _normalize(action_sum)
            i += 1

    val = explore(env, policy, render_mode=render_mode)

    pp.choice(elements=[1, 0], p=[val, env.reward_range[1] - val], name='r')

    return val


def main(env_id, interval, samples, bias, enable_progress=False):
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

    def model_fn(pp):
        return model(pp, env, actions, bias)

    driver = InferenceDriver(model=model_fn, enable_progress=enable_progress)

    driver.condition(label='r-0', value=1)
    driver.init_model()

    # if bias is None:
    #     # learned bias
    #     driver.prior(label="bias-0", value=.25)
    #     driver.prior(label="bias-1", value=.25)
    #     driver.prior(label="bias-2", value=.25)
    #     driver.prior(label="bias-3", value=.25)

    driver.burn_in(steps=100)

    policy_shape = (*env.observation_space.shape, env.action_space.n)
    posterior_policy = np.zeros(policy_shape, dtype=np.int8)
    trace = driver.run_inference(interval=interval, samples=samples)
    for k, v in trace.items():
        if 'policy' in k:
            _, y, x, _ = k.split('_')
            posterior_policy[int(y), int(x), int(v['value'])] = 1

    results = {
        'trace': trace,
        'policy': posterior_policy,
        'rewards': driver.model_results,
        'samples': driver.samples,
        'log_likelihoods': driver.lls,
    }

    bias_name = 'learned' if bias is None else '-'.join(str(b) for b in bias)
    name = '_'.join([env_id, str(interval), str(samples), bias_name])
    path = 'results/{}.pkl'.format(name)
    with open(path, 'wb') as f:
        pickle.dump(results, f)

    return path


if __name__ == '__main__':
    # env_id = 'OpenMazeUrgencyMax10CyclesInWindowSize2-v0'
    # env_id = 'OpenMazeUrgencyMax4CyclesInWindowSize2-v0'
    # env_ids = ['OpenMazeDiscountCompletion-v0']
    # env_ids = ['OpenMazeOnlyCompletionReward-v0']
    env_ids = ['OpenMazeDiscountCompletion-v0']
    intervals = [1]
    samples = range(200, 2001, 100)
    biases = [None, [.25] * 4, [.7, .1, .1, .1], [.1, .1, .7, .1]]

    inputs = itertools.product(env_ids, intervals, samples, biases)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as p:
        for filename in p.starmap(main, inputs):
            print(filename)
