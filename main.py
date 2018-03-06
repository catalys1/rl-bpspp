import gym
import gym_openmaze
import numpy as np
from tqdm import trange
import scipy.stats


class Model:
    def __init__(self, env, bias=None):
        self.env = env

        if bias is None:
            self._bias = scipy.stats.dirichlet([1.0] * env.action_space.n)
            self.bias = self._bias.rvs()[0]
        else:
            self._bias = None
            self.bias = bias

        self._policy = scipy.stats.multinomial(n=1, p=self.bias)
        self.policy = self._policy.rvs(size=env.observation_space.shape)

        self.reward = 0.0
        self.log_likelihood = np.log(self.reward)

    def propose_policy(self, from_policy=None, num_states_to_change=1):
        if from_policy is None:
            from_policy = self.policy

        if self._bias is not None and np.random.randint(0, 2) == 0:
            # TODO: this doesnt seem right? just sample it again?
            self._bias = scipy.stats.dirichlet([1.0] * self.env.action_space.n)
            self.bias = self._bias.rvs()[0]
        else:
            from_policy = np.copy(from_policy)
            for i in range(num_states_to_change):
                state_row = np.random.randint(0, from_policy.shape[0])
                state_col = np.random.randint(0, from_policy.shape[1])
                action = np.random.randint(0, self.env.action_space.n)
                distribution = np.zeros(self.env.action_space.n)
                distribution[action] = 1.0
                from_policy[state_row, state_col] = distribution

        return from_policy

    def run(self, policy=None, render_mode=None):
        if policy is None:
            policy = self.policy

        state = self.env.reset()
        total_reward = 0.0
        while True:
            action = np.random.choice(self.env.action_space.n, p=policy[tuple(state)])
            observation, reward, done, info = self.env.step(action)

            if render_mode is not None:
                self.env.render(mode=render_mode)

            total_reward += reward

            if done:
                break
            state = observation

        return total_reward

    def compute_log_likelihood(self, policy=None, reward=None):
        if policy is None:
            policy = self.policy
        if reward is None:
            reward = self.reward

        ll = 0.0 if self._bias is None else self._bias.logpdf(self.bias)
        ll += np.log(self.bias[policy.argmax(axis=-1).flatten()]).sum()
        ll += reward
        return ll


def metropolis_hastings(model, iterations=1000):
    acceptance_count = 0
    with trange(iterations) as progress:
        progress.set_description('a: inf, acceptance count: 0')
        for _ in progress:
            policy_prime = model.propose_policy()
            reward_prime = model.run(policy=policy_prime)

            log_likelihood_prime = model.compute_log_likelihood(policy=policy_prime, reward=reward_prime)
            a = log_likelihood_prime - model.log_likelihood
            if np.log(np.random.rand()) <= np.minimum(0.0, a):
                model.policy = policy_prime
                model.reward = reward_prime
                model.log_likelihood = log_likelihood_prime
                acceptance_count += 1
                progress.set_description('a: {}, acceptance count: {}'.format(a, acceptance_count))
    return model


def main(search_for_policy=metropolis_hastings):
    gym.envs.registration.register(
        id='openmaze-custom-v0',
        entry_point='gym_openmaze.envs.openmaze:OpenMaze',
        kwargs={'size': (10, 7), 'random': False},
        max_episode_steps=10000
    )
    env = gym.make('openmaze-custom-v0')  # TODO: change to random

    # model = Model(env, bias=[.85, .05, .05, .05])
    model = Model(env, bias=None)
    search_for_policy(model)
    model.run(render_mode='human')


if __name__ == '__main__':
    main()
