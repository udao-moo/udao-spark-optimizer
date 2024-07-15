# Copyright (c) 2024 École Polytechnique
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: TODO
#
# Created at 05/02/2024


from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from udao.optimization.concepts import (
    EnumVariable,
    FloatVariable,
    IntegerVariable,
    Variable,
)

# from utils.variable import (
#     EnumVariable,
#     FloatVariable,
#     IntegerVariable,
#     Variable,
# )


class RandomSampler:
    @dataclass
    class Params:
        n_samples_per_param: int
        "the number of samples per variable"

        seed: Optional[int] = None
        "random seed for generatino of samples"

    def __init__(self, params: Params) -> None:
        """
        :param params: RandomSampler.Params
        """
        super().__init__()
        self.n_samples_per_param = params.n_samples_per_param
        self.seed = params.seed

    def _process_variable(self, var: Variable) -> np.ndarray:
        """Generate samples of a variable"""
        if isinstance(var, FloatVariable):
            return np.random.uniform(var.lower, var.upper, self.n_samples_per_param)
        elif isinstance(var, IntegerVariable):
            return np.random.randint(
                var.lower, var.upper + 1, size=self.n_samples_per_param
            )
        elif isinstance(var, EnumVariable):
            inds = np.random.randint(0, len(var.values), size=self.n_samples_per_param)
            return np.array(var.values)[inds]
        else:
            raise NotImplementedError(
                f"ERROR: variable type {type(var)} is not supported!"
            )

    def _get_input(self, variables: List[Variable]) -> np.ndarray:
        """
        generate samples of variables

        Parameters:
        -----------
        variables: List[Variable],
            lower and upper var_ranges of variables(non-ENUM),
            and values of ENUM variables
        Returns:
        np.ndarray,
            variables (n_samples * n_vars)
        """
        n_vars = len(variables)
        x = np.zeros([self.n_samples_per_param, n_vars])
        np.random.seed(self.seed)
        for i, var in enumerate(variables):
            x[:, i] = self._process_variable(var)
        return x
