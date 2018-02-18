import pprint
import time
from collections import defaultdict
from enum import Enum
from typing import Callable, Tuple

import gym
import numpy as np
from tqdm import tqdm, trange


class VisitType(Enum):
    """Monte Carlo prediction type of visiting a state (s).
    See Also: "Reinforcement Learning: An Introduction", Richard S. Sutton and Andrew G. Barto, p. 74
    """
    FIRST = 1  # first-visit MC method estimates vπ(s) as the average of the returns following first visits to s
    EVERY = 2  # every-visit MC method averages the returns following all visits to s


def run_maze(env, policy: Callable[[np.array], np.array]) -> Tuple[np.array, np.array, float]:
    state = env.reset()
    while True:
        probabilities = policy(state)
        action = np.random.choice(len(probabilities), p=probabilities)
        observation, reward, done, info = env.step(action)
        if done:
            break
        state = observation
        yield observation, action, reward


def create_epsilon_greedy_policy(q: defaultdict, epsilon: float, num_actions: int) -> Callable[[np.array], np.array]:
    """Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        q: A dictionary that maps state -> action-values. Each value is a np.array of length number_of_actions.
        epsilon: The probability of sampling a random action. Float between 0 and 1.
        num_actions: Number of actions in the environment.

    Returns:
        A policy function which maps an observation -> action probabilities, in the form of a np.array of length
        number_of_actions.
    """

    def policy_fn(observation: np.array) -> np.array:
        action_probabilities = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(q[observation])
        action_probabilities[best_action] += (1.0 - epsilon)
        return action_probabilities

    return policy_fn


def monte_carlo(env,
                create_policy=create_epsilon_greedy_policy, explore=run_maze,
                num_episodes=100, discount_factor=.8, epsilon=lambda i: 0.1 ** i,
                visit_type=VisitType.EVERY, debug=False) -> Tuple[defaultdict, Callable[[np.array], np.array]]:
    """Monte Carlo Control using Epsilon-Greedy policies. Finds an optimal epsilon-greedy policy.

    Args:
        env: An OpenAI gym environment.
        create_policy: A function used to create a policy function which maps an observation -> action probabilities.
        num_episodes: Number of episodes to sample.
        explore: A function used to explore the env.
        discount_factor: Gamma discount factor for future rewards.
        epsilon: A function to compute the probability of sampling a random action.
        visit_type: The Monte-Carlo visit type (FIST, EVERY).
        debug: Whether or not to enable debug print statements.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        Policy is a function which maps an observation -> action probabilities.
    """
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sums = defaultdict(float)
    returns_counts = defaultdict(int)
    policy = lambda: None

    with trange(num_episodes) as episodes:
        for e in episodes:
            policy = create_policy(q, epsilon(e), env.action_space.n)

            episode = []
            for observation, action, reward in explore(env, policy):
                if debug:
                    env.render()
                episode.append((observation, action, reward))

            v = 0
            for i, (state, action, reward) in reversed(list(enumerate(episode[:-1]))):

                if visit_type == VisitType.FIRST and any(e for e in episode[:i - 1] if e[0] == state):
                    continue

                v += episode[i + 1][2] * discount_factor ** i
                returns_sums[(state, action)] += v
                returns_counts[(state, action)] += 1
                q[state][action] = returns_sums[(state, action)] / returns_counts[(state, action)]

            episodes.set_description(str(env._elapsed_steps))
            if debug:
                tqdm.write(pprint.pformat(dict(q)))

    return q, policy


def main(control=monte_carlo, debug=False):
    from gym.envs.registration import register
    register(
        id='NChain-custom-v0',
        entry_point='gym.envs.toy_text:NChainEnv',
        kwargs={'n': 4, 'slip': 0.0, 'small': 0, 'large': 10},
        max_episode_steps=100,
    )
    env = gym.make("NChain-custom-v0")

    v = defaultdict(float)
    q, policy = control(env, create_epsilon_greedy_policy)
    for state, actions in q.items():
        action_value = np.max(actions)
        v[state] = action_value

    time.sleep(.1)  # so print output isn't messed up
    for k in sorted(v, key=v.get, reverse=True):
        print(k, v[k])


if __name__ == '__main__':
    main()
