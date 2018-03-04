import gym
import numpy as np
from tqdm import trange
import scipy.stats


class Model:
    def __init__(self, env):
        self.env = env

        # TODO: do these need to change throughout the program?
        self.bias = scipy.stats.dirichlet([1.0] * env.action_space.n)
        self.pi = scipy.stats.multinomial(n=1, p=self.bias.rvs()[0])

        if isinstance(env.observation_space, gym.spaces.Box):
            self._policy_size = self.env.observation_space.high[0] + 1
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            self._policy_size = env.observation_space.n
        else:
            raise ValueError("Cannot determine policy size")

        self.policy = self.pi.rvs(size=self._policy_size)

        self.reward = self.run()

    def propose_policy(self, from_policy=None):
        if from_policy is None:
            from_policy = self.policy

        if np.random.randint(0, 2) == 0:
            # TODO: this doesnt seem right? just sample it again?
            self.bias = scipy.stats.dirichlet([1.0] * self.env.action_space.n)
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
            # probabilities = policy(state)
            # action = np.random.choice(len(probabilities), p=probabilities)
            action = np.argwhere(policy[state])[0][0]  # TODO: awk...
            observation, reward, done, info = self.env.step(action)

            if render:
                self.env.render()

            total_reward += reward

            if done:
                break
            state = observation

        return total_reward

    def distributions(self):
        return [self.bias, self.pi]


def metropolis_hastings(model, iterations=1000):
    # acceptance_count = 0
    log_liklihood = 0.0
    for _ in trange(iterations):
        policy_prime = model.propose_policy()
        reward_prime = model.run(policy=policy_prime)
        # TODO: need to get the logpdf of the dirichlet at the current frozen dirichlet sample and then input that into logpdf of multinomial
        for dist in model.distributions():
            log_liklihood += dist.logpdf()
        log_liklihood += reward_prime
        a = log_liklihood - np.log(model.reward)
        if np.log(np.random.rand()) <= np.minimum(1, a):
            model.policy = policy_prime
            model.reward = reward_prime
            # acceptance_count += 1
    return model


def main(search_for_policy=metropolis_hastings):
    # env = gym.make("maze-sample-5x5-v0")  # TODO: change to random
    gym.envs.registration.register(
        id='NChain-custom-v0',
        entry_point='gym.envs.toy_text:NChainEnv',
        kwargs={'n': 5, 'slip': 0.0},
        max_episode_steps=1000
    )
    env = gym.make('NChain-custom-v0')
    model = Model(env)
    model = search_for_policy(model)
    print(model.policy)


if __name__ == '__main__':
    main()
