import numpy as np
import random
import tqdm
import gym
import gym_maze
import pprint as pp
from collections import defaultdict


# def random_policy(env):
#     return env.action_space.sample()


def run_maze(env, policy, limit=100000):
    state = env.reset()
    for _ in range(limit):
        probs = policy(state)
        action = np.random.choice(len(probs), p=probs)
        observation, reward, done, info = env.step(action)
        if done:
            break
        state = observation
        yield observation, action, reward


def make_epsilon_greedy_policy(q, epsilon, number_of_actions):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        number_of_actions: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation):
        a = np.ones(number_of_actions, dtype=float) * epsilon / number_of_actions
        best_action = np.argmax(q[tuple(observation)])
        a[best_action] += (1.0 - epsilon)
        return a

    return policy_fn


def monte_carlo(env, num_episodes=10, discount_factor=.8, epsilon=0.1, explore=run_maze, visit_type='first'):
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sums = defaultdict(float)
    returns_counts = defaultdict(int)

    with tqdm.trange(num_episodes) as episodes:
        for e in episodes:

            policy = make_epsilon_greedy_policy(q, epsilon ** e, env.action_space.n)

            episode = []
            for observation, action, reward in explore(env, policy):
                # env.render()
                episode.append((tuple(observation), action, reward))

            v = 0
            iterations_to_complete = 0
            for i, (state, action, reward) in reversed(list(enumerate(episode[:-1]))):

                if visit_type == 'fist' and (state, action, reward) in episode[:i - 1]:
                    continue

                v += episode[i + 1][2]
                returns_sums[(state, action)] += v
                returns_counts[(state, action)] += 1
                avg = returns_sums[(state, action)] / returns_counts[(state, action)]
                q[state][action] = avg
                iterations_to_complete += 1

            episodes.set_description(str(iterations_to_complete))

    return q, policy


def mc_control_epsilon_greedy(env, num_episodes=1000, discount_factor=1.0, epsilon=0.1, explore=run_maze, visit_type='first'):
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sums = defaultdict(float)
    returns_count = defaultdict(int)

    with tqdm.trange(num_episodes) as episodes:
        for e in episodes:

            policy = make_epsilon_greedy_policy(q, epsilon ** e, env.action_space.n)

            episode = []
            sa_in_episode = set()
            for observation, action, reward in explore(env, policy):
                if e == 0:
                    env.render()
                episode.append((tuple(observation), action, reward))
                sa_in_episode.add((tuple(observation), action))

            # v = 0
            for state, action in sa_in_episode:
                sa_pair = (state, action)
                # Find the first occurance of the (state, action) pair in the episode
                first_occurence_idx = next(i for i, x in enumerate(episode)
                                           if np.array_equal(x[0], state) and x[1] == action)
                # Sum up all rewards since the first occurance
                G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
                # Calculate average return for this state over all sampled episodes
                returns_sums[sa_pair] += G
                returns_count[sa_pair] += 1
                q[state][action] = returns_sums[sa_pair] / returns_count[sa_pair]

            pp.pprint(q)
            episodes.set_description(str(len(episode)))

    return q, policy


def main(control=mc_control_epsilon_greedy):
    env = gym.make("maze-sample-5x5-v0")  # TODO: change to random
    v = defaultdict(float)
    q, policy = control(env)
    for state, actions in q.items():
        action_value = np.max(actions)
        v[state] = action_value

    for k in sorted(v, key=v.get, reverse=True):
        print(k, v[k])


if __name__ == '__main__':
    main()
