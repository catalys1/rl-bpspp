from os.path import splitext, basename
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

plot_data = {
    '0.1-0.1-0.7-0.1': {'label': 'South bias', 'data': {'x': [], 'y': []}, 'color': 'black'},
    '0.25-0.25-0.25-0.25': {'label': 'No bias', 'data': {'x': [], 'y': []}, 'color': 'blue'},
    '0.7-0.1-0.1-0.1': {'label': 'North bias', 'data': {'x': [], 'y': []}, 'color': 'red'},
    'learned': {'label': 'Learned bias', 'data': {'x': [], 'y': []}, 'color': 'green'},
}  # yapf: disable

args = [(p, splitext(basename(p))[0].split('_')) for p in sys.argv[1:]]
files = sorted(args, key=lambda x: (int(x[1][2]), x[1][3]))

title = None
for filepath, (env_id, _, samples, bias) in files:
    if title is None:
        title = env_id.split('-')[0]
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        plot_data[bias]['data']['x'].append(int(samples))
        plot_data[bias]['data']['y'].append(np.sum(data['rewards']))
    print(env_id, samples, bias)

plt.figure(figsize=(5, 3))
for _, plot in plot_data.items():
    plt.ylabel('Accumulated reward', fontsize=10)
    plt.xlabel('Policy evaluations', fontsize=10)
    # plt.plot(**plot)  # this should work but matplotlib is dumb
    plt.plot(
        plot['data']['x'],
        plot['data']['y'],
        label=plot['label'],
        color=plot['color'])
plt.legend()
plt.tight_layout()
plt.savefig('charts_creative/{}.png'.format(title))
