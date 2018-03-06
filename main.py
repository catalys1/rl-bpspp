import gym
import gym_openmaze
import numpy as np
from tqdm import trange
import scipy.stats


class Model:
    def __init__(self, env, render_mode=None):
        self.env = env

        # TODO: do these need to change throughout the program?
        self._bias = scipy.stats.dirichlet([1.0] * env.action_space.n)
        self.bias = self._bias.rvs()[0]
        self._policy = scipy.stats.multinomial(n=1, p=self.bias)
        self.policy = self._policy.rvs(size=env.observation_space.shape)

        self.render_mode = render_mode
        self.reward = self.run(render_mode=self.render_mode)

    def propose_policy(self, from_policy=None):
        if from_policy is None:
            from_policy = self.policy

        if np.random.randint(0, 2) == 0:
            # TODO: this doesnt seem right? just sample it again?
            self._bias = scipy.stats.dirichlet([1.0] * self.env.action_space.n)
            self.bias = self._bias.rvs()[0]
        else:
            from_policy = np.copy(from_policy)
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

        if render_mode is None:
            render_mode = self.render_mode

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

    def likelihood(self, policy=None, reward=None):
        if policy is None:
            policy = self.policy
        if reward is None:
            reward = self.reward

        ll = self._bias.pdf(self.bias)
        # TODO: do we have to loop?
        for p in policy:
            for x in p:
                ll += self.bias[np.argmax(x)]
        ll += reward
        return ll


def metropolis_hastings(model, iterations=1000):
    with trange(iterations) as progress:
        for _ in progress:
            policy_prime = model.propose_policy()
            reward_prime = model.run(policy=policy_prime)

            a = np.log(model.likelihood(policy=policy_prime, reward=reward_prime)) - model.reward
            if np.log(np.random.rand()) <= np.minimum(1, a):
                model.policy = policy_prime
                model.reward = reward_prime
            progress.set_description('reward {}, a {}'.format(model.reward, a))
    return model


def main(search_for_policy=metropolis_hastings):
    env = gym.make("openmaze-v0")  # TODO: change to random

    model = Model(env, render_mode='human')
    start_policy = model.policy
    model = search_for_policy(model)
    print(np.abs(model.policy - start_policy).sum())


if __name__ == '__main__':
    main()
