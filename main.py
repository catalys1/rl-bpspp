from typing import Callable, Tuple

import gym
import numpy as np
from tqdm import trange
from collections import defaultdict

def run_maze(env, policy: Callable[[np.array], np.array], render=False) -> Tuple[np.array, np.array, float]:
    state = env.reset()
    while True:
        # probabilities = policy(state)
        # action = np.random.choice(len(probabilities), p=probabilities)
        observation, reward, done, info = env.step(action)

        if render:
            env.render()

        yield state, action, reward

        if done:
            return
        state = observation


def policy_proposal_kernel(parameters):
    parameter_key = np.random.choice(parameters.keys())
    if parameter_key == 'bias':
        new_bias = np.random.dirichlet([1.0]*4)
        parameters[parameter_key] = new_bias
    elif parameter_key == 'policy':
        policy = parameters[parameter_key]
        state_row = np.random.randint(0, policy.shape[0])
        state_col = np.random.randint(0, policy.shape[1])
        action = np.random.randint(0, 4)
        distribution = np.zeros(4)
        distribution[action] = 1.0
        policy[state_row, state_col] = distribution
        parameters[parameter_key] = policy
    return parameters


def value(env, policy, explore=run_maze):
    total_reward = 0.0
    for state, action, reward in trange(explore(env, policy)):
        total_reward += reward
    return total_reward


def init_parameters(rows=3, cols=3):
    bias = np.random.dirichlet(1.0, 4)
    pi = np.random.multinomial(1, bias, (rows, cols))
    return {'bias': bias, 'policy': pi}


def metropolis_hastings(env, current_policy, current_value, value_fn=value, proposal_kernel_fn=policy_proposal_kernel,
                        iterations=1000):
    # acceptance_count = 0
    for _ in trange(iterations):
        policy_prime = proposal_kernel_fn(current_policy)
        value_prime = value_fn(env, policy_prime)
        # gaussian for proposal distribution; q(x|x') / q(x'|x) == 1.0
        a = value_prime / current_value
        if np.random.random() <= np.minimum(1, a):
            current_policy = policy_prime
            current_value = value_prime
            # acceptance_count += 1
#     return current_policy, current_value


def run_model():

    pass

def main(search_for_policy=metropolis_hastings):
    gym.envs.registration.register(
        id='NChain-custom-v0',
        entry_point='gym.envs.toy_text:NChainEnv',
        kwargs={'n': 5, 'slip': 0.0},
        max_episode_steps=1000
    )
    env = gym.make('NChain-custom-v0')

    parameters = init_parameters()
    current_value = value(env, parameters['policy'])
    current_trace = 0






    # env = gym.make("maze-sample-3x3-v0")  # TODO: change to random




    # v = defaultdict(float)
    # q, policy = control(env, create_epsilon_greedy_policy)
    # for state, actions in q.items():
    #     action_value = np.max(actions)
    #     v[state] = action_value
    #
    # time.sleep(.1)  # so print output isn't messed up
    # for k in sorted(v, key=v.get, reverse=True):
    #     print(tuple(_to_unhashable(k, dtype=int)), v[k])

if __name__ == '__main__':
    main()
