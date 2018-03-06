import gym
import gym_openmaze
import numpy as np
from tqdm import trange
import scipy.stats


class Model:
    def __init__(self, env):
        self.env = env

        # TODO: do these need to change throughout the program?
        self._bias = scipy.stats.dirichlet([1.0] * env.action_space.n)
        self.bias = self._bias.rvs()[0]
        self._policy = scipy.stats.multinomial(n=1, p=self.bias)
        self.policy = self._policy.rvs(size=env.observation_space.shape)

        self.reward = self.run(render=False)

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

    def run(self, policy=None, render=False):
        if policy is None:
            policy = self.policy

        state = self.env.reset()
        total_reward = 0.0
        while True:
            action = np.random.choice(self.env.action_space.n, p=policy[tuple(state)])
            observation, reward, done, info = self.env.step(action)

            if render:
                self.env.render()

            total_reward += reward

            if done:
                break
            state = observation

        return total_reward

    def log_likelihood(self, policy=None, reward=None):
        if policy is None:
            policy = self.policy
        if reward is None:
            reward = self.reward

        ll = self._bias.logpdf(self.bias)
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
            reward_prime = model.run(policy=policy_prime, render=False)

            a = model.log_likelihood(policy=policy_prime, reward=reward_prime) - np.log(model.reward)
            if np.log(np.random.rand()) <= np.minimum(1, a):
                model.policy = policy_prime
                model.reward = reward_prime
            progress.set_description('reward {}'.format(model.reward))
    return model


def main(search_for_policy=metropolis_hastings):
    env = gym.make("openmaze-v0")  # TODO: change to random
    # gym.envs.registration.register(
    #     id='NChain-custom-v0',
    #     entry_point='gym.envs.toy_text:NChainEnv',
    #     kwargs={'n': 5, 'slip': 0.0},
    #     max_episode_steps=1000
    # )
    # env = gym.make('NChain-custom-v0')
    model = Model(env)
    start_policy = model.policy
    model = search_for_policy(model)
    print(np.abs(model.policy - start_policy).sum())


if __name__ == '__main__':
    main()
