import math
import operator
from functools import reduce

import numpy as np
import scipy.stats
import scipy.special
import inspect
from tracetable import TraceTable


class ProbPy:
    def __init__(self):
        self.table = TraceTable()

    # ----------------------------------
    # Trace Functions

    def accept_proposed_trace(self):
        self.table.accept_proposed_trace()

    def propose_new_trace(self):
        self.table.propose_new_trace()

    def pick_random_erp(self):
        return self.table.pick_random_erp()

    def store_new_erp(self, label, value, erp, parameters):
        likelihood = self._get_likelihood(erp, parameters, value)
        self.table.add_entry_to_proposal(
            label, value, erp, parameters, likelihood, True
        )

    def score_current_trace(self):
        return self.table.score_current_trace()

    def score_proposed_trace(self):
        self.table.collect_ll_stale()
        self.table.collect_ll()
        return self.table.get_lls()

    # ----------------------------------
    # Primitives

    def random(self, name, size=None, loop_iter=0):
        label = self._get_label(name, loop_iter)
        value = self.table.read_entry_from_proposal(label, "random", None)
        if type(value) == bool:
            add_fresh = value
            value = np.random.random(size=size)
        else:
            add_fresh = False
        likelihood = self._uniform_pdf(0, 1)
        parameters = {}
        self.table.add_entry_to_proposal(
            label, value, "random", parameters, likelihood, add_fresh
        )
        return value

    def randint(self, name, low, high=None, size=None, loop_iter=0):
        label = self._get_label(name, loop_iter)
        value = self.table.read_entry_from_proposal(label, "randint", None)
        if type(value) == bool:
            add_fresh = value
            value = np.random.randint(low=low, high=high, size=size)
        else:
            add_fresh = False
        likelihood = self._uniform_pdf(low, high)
        parameters = {"low": low, "high": high, "size": size}
        self.table.add_entry_to_proposal(
            label, value, "randint", parameters, likelihood, add_fresh
        )
        return value

    def normal(self, name, loc=0.0, scale=1.0, size=None, loop_iter=0):
        label = self._get_label(name, loop_iter)
        value = self.table.read_entry_from_proposal(label, "normal", None)
        if type(value) == bool:
            add_fresh = value
            value = np.random.normal(loc=loc, scale=scale, size=size)
        else:
            add_fresh = False
        likelihood = self._normal_pdf(loc, scale, value)
        parameters = {"loc": loc, "scale": scale, "size": size}
        self.table.add_entry_to_proposal(
            label, value, "normal", parameters, likelihood, add_fresh
        )
        return value

    def choice(
        self, name, elements, size=None, replace=True, p=None, loop_iter=0
    ):
        label = self._get_label(name, loop_iter)
        value = self.table.read_entry_from_proposal(label, "choice", None)
        if type(value) == bool:
            add_fresh = value
            value = np.random.choice(
                a=elements, size=size, replace=replace, p=p
            )
        else:
            add_fresh = False
        likelihood = self._categorical_pdf(elements, p, value)
        parameters = {
            "elements": elements,
            "size": size,
            "replace": replace,
            "p": p
        }
        self.table.add_entry_to_proposal(
            label, value, "choice", parameters, likelihood, add_fresh
        )
        return value

    # ----------------------------------
    # Probability Density Functions

    def _uniform_pdf(self, low, high):
        if high == None:
            high = low
            low = 0.0
        return -np.log(high - low)

    def _normal_pdf(self, mean, standard_dev, value):
        first_term = -1.0 * np.log(np.sqrt(2.0 * np.pi * (standard_dev**2)))
        second_term = (-1.0 * (value - mean)**2) / (2.0 * (standard_dev**2))
        return first_term + second_term

    def _categorical_pdf(self, elements, p, value):
        # if p == None:
        if p is None:
            num_of_elements = len(elements)
            pdf = 1.0 / num_of_elements
        else:
            pdf = 0.0
            for i in range(len(elements)):
                if elements[i] == value:
                    pdf += p[i]
        return np.log(pdf)

    # ----------------------------------
    # Propsal Kernals

    # x's are independent
    def sample_erp(self, erp, parameters):
        execute_string = "np.random." + erp + "("
        for key in parameters.keys():
            execute_string += key + "=" + str(parameters[key]) + ", "
        if len(parameters.keys()) > 0:
            execute_string = execute_string[:-2]
        execute_string += ")"
        x = eval(execute_string)
        return x, 0, 0

    # x's are dependent
    def simple_proposal_kernal(self, old_value):
        # old_value = np.argmax(old_value)  # TODO: ????
        var = .9
        new_value = np.random.normal(loc=old_value, scale=var)
        F = self._normal_pdf(old_value, var, new_value)
        R = self._normal_pdf(new_value, var, old_value)
        return new_value, F, R

    def choice_proposal_kernal(self, old_value, elements, p):
        new_value = np.random.choice(a=elements, p=p)
        F = self._categorical_pdf(elements, p, new_value)
        R = self._categorical_pdf(elements, p, old_value)
        return new_value, F, R

    # ----------------------------------
    # Helpers

    # deprecated
    def _get_line_label(self, loop_iter):
        stack = inspect.stack()
        caller_stack = stack[2]
        label = str(caller_stack[2]) + "-" + str(loop_iter)
        return label

    def _get_label(self, name, loop_iter):
        return name + "-" + str(loop_iter)

    # TODO: find a cleaner way to do this
    def _get_likelihood(self, erp, parameters, value):
        if erp == "random":
            return self._uniform_pdf(0, 1)
        elif erp == "randint":
            return self._uniform_pdf(parameters["low"], parameters["high"])
        elif erp == "normal":
            return self._normal_pdf(
                parameters["loc"], parameters["scale"], value
            )
        elif erp == "choice":
            return self._categorical_pdf(
                parameters["elements"], parameters["p"], value
            )
