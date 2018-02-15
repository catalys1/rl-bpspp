import random
import tqdm
import numpy as np
import pprint as pp
import envs.maze
from collections import defaultdict


def random_policy(state, available_actions):
    return random.choice(available_actions)


def run_maze(env, obs, policy, all_actions={'n', 's', 'e', 'w'}):
    done = None
    while not done:
        act = policy(obs, list(all_actions - obs))
        obs, done, reward, info = env.step(act)
        yield obs, act, reward


def monte_carlo(env, policy, num_episodes=10, discount_factor=.99):
    value = defaultdict(float)
    returns = defaultdict(list)

    for _ in tqdm.trange(num_episodes):
        state = env.reset()
        episode = []
        for obs, act, reward in run_maze(env, state, policy):
            episode.append((tuple(obs), act, reward))
            state = obs

        v = 0
        for i, (state, act, reward) in reversed(list(enumerate(episode))):
            v += reward * (discount_factor**i)

            # first-visit monte carlo
            if state not in episode[:i]:
                returns[state].append(v)
                value[state] = np.mean(returns[state])

    return value


def main(value=monte_carlo):
    v = value(envs.maze.MazeEnv(None), random_policy)
    pp.pprint(dict(v))


if __name__ == '__main__':
    main()
