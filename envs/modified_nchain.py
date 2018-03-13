import gym
from gym import spaces
from gym.utils import seeding

class ModifiedNChainEnv(gym.Env):
    def __init__(self, n=5, slip=0.0, intermediate=0, end=1):
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.intermediate = intermediate  # payout for 'backwards' action
        self.end = end  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.reward_range = [intermediate, end]
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        if self.np_random.rand() < self.slip:
            action = not action  # agent slipped, reverse action taken
        if action:  # 'backwards'
            reward = self.intermediate
            self.state -= 1
            if self.state < 0:
                self.state = 0
        elif self.state < self.n - 1:  # 'forwards': go up along the chain
            reward = self.intermediate
            self.state += 1
        else:  # 'forwards'
            reward = self.end
            done = True
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state
