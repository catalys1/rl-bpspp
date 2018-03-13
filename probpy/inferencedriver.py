import copy

import numpy as np

import matplotlib.pyplot as plt
from probpy.probpy import ProbPy
from tqdm import trange


class InferenceDriver:
    def __init__(self, model, enable_progress=True):
        self.pp = ProbPy()
        self.model = model
        self.model_results = []
        self.samples = []
        self.lls = []
        self.enable_progress = enable_progress

    def init_model(self):
        # prime the database
        self.model(self.pp)
        self.pp.accept_proposed_trace()

    def burn_in(self, steps):
        disable = not self.enable_progress
        for i in trange(steps, desc='  burn in', disable=disable):
            self.inference_step()

    def run_inference(self, interval, samples):
        self.num_samples = samples
        num_accepted = 0
        model_success = 0
        threshold = None
        d = not self.enable_progress
        with trange(samples, desc='inference', disable=d) as progress_bar:
            for s in progress_bar:
                for i in range(interval):
                    did_accept, threshold, model_result = self.inference_step()
                    num_accepted += did_accept
                    self.model_results.append(model_result)
                self.samples.append(copy.deepcopy(self.pp.table.trace))
                if np.isclose(model_result, 1.):
                    model_success += 1  # TODO: remove, this is model specific
                progress_bar.set_postfix(
                    acceptance_count=num_accepted,
                    acceptance_rate=num_accepted / max(1, s),
                    model_success=model_success)
        return self.pp.table.trace

    def inference_step(self):
        # score the current trace
        ll = self.pp.score_current_trace()
        self.lls.append(ll)
        # start new trace (copy of the old trace)
        self.pp.propose_new_trace()
        # pick a random ERP
        label, entry = self.pp.pick_random_erp()
        # propose a new value
        if entry["erp"] == "choice":
            value, F, R = self.pp.choice_proposal_kernal(
                entry["value"], entry["parameters"]["elements"],
                entry["parameters"]["p"])
        else:
            value, F, R = self.pp.simple_proposal_kernal(entry["value"])
        # value, F, R = self.pp.sample_erp(entry["erp"], entry["parameters"]) # sample kernal
        self.pp.store_new_erp(label, value, entry["erp"], entry["parameters"])
        # re-run the model
        model_result = self.model(self.pp)
        # score the new trace
        ll_prime, ll_fresh, ll_stale = self.pp.score_proposed_trace()
        # calculate MH acceptance ratio
        threshold = ll_prime - ll + R - F + ll_stale - ll_fresh
        # accept or reject
        if np.log(np.random.rand()) < threshold:
            self.pp.accept_proposed_trace()
            return 1, threshold, model_result
        return 0, threshold, model_result

    def condition(self, label, value):
        self.pp.table.condition(label, value)

    def prior(self, label, value):
        self.pp.table.prior(label, value)

    #
    # def return_traces(self):
    #     return self.samples

    def finalize(self, fn=None):
        if fn is None:
            fn = self.model
        return fn(self.pp)

    # def return_model_results(self):
    #     return self.model_results

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
                for key, item in s.items():
                    if k + '-0' == key:
                        data[k].append(item['value'])
        return data

    def plot_ll(self, name="ll_figure.png"):
        plt.figure()
        plt.ylabel('Log likelihood', fontsize=10)
        plt.xlabel('Episodes', fontsize=10)
        plt.plot(self.lls)
        plt.savefig(name)

    def plot_model_results(self, mn):
        plt.figure()
        plt.plot(self.model_results)
        plt.savefig()
        return self.model_results
