# Reinforcement Learning Using Bayesian Policy Search with Policy Priors

## Maze Environment

You can run a simple random-action simulation with the following code:

```
from maze import MazeEnv
import random

env = MazeEnv(None)

actions = {'n','s','e','w'}

obs = env.reset()
for i in range(10000):
    act = random.choice(list(actions-obs))
    obs,done,reward,info = env.step(act)
    if done:
        print(f'Finished in {i} steps')
        break
print(f'Failed to finish in {i} steps')
```
