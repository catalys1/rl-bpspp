import copy

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from probpy import ProbPy
from tqdm import trange


class InferenceDriver:

    def __init__(self, model):
        self.pp = ProbPy()
        self.model = model
        self.samples = []
        self.lls = []

    def init_model(self):
        # prime the database
        self.model(self.pp)
        self.pp.accept_proposed_trace()

    def burn_in(self, steps):
        for i in trange(steps, desc='  burn in'):
            self.inference_step()

    def run_inference(self, interval, samples):
        self.num_samples = samples
        num_accepted = 0
        threshold = None
        with trange(samples, desc='inference') as progress_bar:
            for s in progress_bar:
                for i in range(interval):
                    did_accept, threshold = self.inference_step(s, progress_bar)
                    num_accepted += 1 if did_accept else 0
                self.samples.append(copy.deepcopy(self.pp.table.trace))
                progress_bar.set_postfix(
                    {'num_accepted': num_accepted,
                     'acceptance_rate': num_accepted / max(1, s),
                     'threshold': threshold})
        return self.pp.table.trace

    def inference_step(self, step_count=0, progress_bar=None):
        # score the current trace
        ll = self.pp.score_current_trace()
        self.lls.append(ll)
        # start new trace (copy of the old trace)
        self.pp.propose_new_trace()
        # pick a random ERP
        label, entry = self.pp.pick_random_erp()
        # propose a new value
        if entry["erp"] == "choice":
            value, F, R = self.pp.choice_proposal_kernal(entry["value"],
                                                         entry["parameters"]["elements"], entry["parameters"]["p"])
        else:
            value, F, R = self.pp.simple_proposal_kernal(entry["value"])
        # value, F, R = self.pp.sample_erp(entry["erp"], entry["parameters"]) # sample kernal
        self.pp.store_new_erp(label, value, entry["erp"], entry["parameters"])
        # re-run the model
        self.model(self.pp, step_count, progress_bar)
        # score the new trace
        ll_prime, ll_fresh, ll_stale = self.pp.score_proposed_trace()
        # calculate MH acceptance ratio
        threshold = ll_prime - ll + R - F + ll_stale - ll_fresh
        # accept or reject
        if np.log(np.random.rand()) < threshold:
            self.pp.accept_proposed_trace()
            return True, threshold
        return False, threshold

    def condition(self, label, value):
        self.pp.table.condition(label, value)

    def prior(self, label, value):
        self.pp.table.prior(label, value)

    def return_traces(self):
        return self.samples

    def return_values(self, keys):
        values = {}
        val_cnt = {}
        for s in self.samples:
            for key, item in s.items():
                if key.split("-")[0] in keys:
                    if key in values:
                        # values[key] = float(values[key] + item["value"]) / 2.0
                        values[key] = float(values[key] + item["value"])
                        val_cnt[key] += 1
                    else:
                        values[key] = item["value"]
                        val_cnt[key] = 1.0
        # return values
        return {k: v / val_cnt[k] for k, v in values.items()}

    def return_string_values(self, keys):
        values = {}
        for s in self.samples:
            for key, item in s.items():
                if key.split("-")[0] in keys:
                    if key in values:
                        values[key].append(item["value"])
                    else:
                        values[key] = [item["value"]]
        return values

    def return_plt_data(self, keys):
        data = {}

        for k in keys:
            data[k] = []
            for s in self.samples:
                values_dict = {}
                for key, item in s.items():
                    if k + '-0' == key:
                        data[k].append(item['value'])
        return data

    def graph_ll(self):
        plt.plot(range(len(self.lls)), self.lls)
        plt.savefig("ll_figure.png")
