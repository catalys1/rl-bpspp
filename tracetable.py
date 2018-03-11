from collections import OrderedDict
from copy import deepcopy

import numpy as np


class TraceTable:
    def __init__(self):
        self.trace = {}  # D from the paper
        self.proposed_trace = {}
        self.ll = 0
        self.ll_fresh = 0
        self.ll_stale = 0
        self.active = set()
        self.clamped = {}

    def add_entry_to_proposal(
        self, label, value, erp, parameters, likelihood, add_fresh
    ):
        # TODO: correct?
        if (add_fresh):
            self.ll_fresh += likelihood
        self.active.add(label)
        self.proposed_trace[label] = {
            "value": value,
            "erp": erp,
            "parameters": parameters,
            "likelihood": likelihood
        }

    def read_entry_from_proposal(self, label, erp, parameters):
        if label in self.clamped.keys():
            return self.clamped[label]
        if label in self.proposed_trace and self.proposed_trace[label][
            "erp"
        ] == erp:
            if self.proposed_trace[label]["parameters"] == parameters:
                self.active.add(label)
                return self.proposed_trace[label]["value"]
            else:
                return False
        else:
            return True

    def pick_random_erp(self):
        keys = list(set(self.trace.keys()) - set(self.clamped.keys()))
        if len(keys) <= 0:
            raise ValueError("The only random variables are clamped...")
        label = keys[np.random.choice(range(len(keys)))]
        return label, self.trace[label]

    def accept_proposed_trace(self):
        self.trace = deepcopy(self.proposed_trace)

    def propose_new_trace(self):
        self.ll = 0
        self.ll_fresh = 0
        self.ll_stale = 0
        self.active = set()
        self.proposed_trace = deepcopy(self.trace)

    def collect_ll_stale(self):
        for label in self.proposed_trace.keys():
            if label not in self.active:
                self.ll_stale += self.proposed_trace[label]["likelihood"]
                del self.proposed_trace[label]

    def collect_ll(self):
        for label in self.proposed_trace.keys():
            self.ll += self.proposed_trace[label]["likelihood"]

    def get_lls(self):
        return self.ll, self.ll_fresh, self.ll_stale

    def score_current_trace(self):
        ll = 0
        for label in self.trace.keys():
            ll += self.trace[label]["likelihood"]
        return ll

    def condition(self, label, value):
        self.clamped[label] = value

    def prior(self, label, value):
        self.trace[label]["value"] = value
