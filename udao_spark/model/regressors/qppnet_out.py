from dataclasses import dataclass

import torch as th
from udao.model import BaseRegressor


class QPPNetOut(BaseRegressor):
    @dataclass
    class Params(BaseRegressor.Params):
        ...

    def __init__(self, net_params: Params) -> None:
        """_summary_"""
        super().__init__(net_params)

    def forward(self, embedding: th.Tensor, inst_feat: th.Tensor) -> th.Tensor:
        return th.cat([embedding] * self.output_dim, dim=1)
