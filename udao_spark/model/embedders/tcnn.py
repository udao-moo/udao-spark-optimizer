from dataclasses import dataclass
from typing import Dict

import dgl
import numpy as np
import torch as th
import torch.nn as nn
from dgl.udf import EdgeBatch, NodeBatch
from udao.model import BaseGraphEmbedder
from udao.model.embedders.graph_transformer import ReadoutType


class TreeConvUnit(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(TreeConvUnit, self).__init__()
        self.W_top = nn.Linear(in_dim, out_dim, bias=False)
        self.W_left = nn.Linear(in_dim, out_dim, bias=False)
        self.W_right = nn.Linear(in_dim, out_dim, bias=False)

    def message_func(self, edges: EdgeBatch) -> Dict[str, th.Tensor]:
        return {"h": edges.src["h"]}

    def reduce_func(self, nodes: NodeBatch) -> Dict[str, th.Tensor]:
        h = nodes.mailbox["h"]  # Shape: (num_nodes, num_neighbors, feature_size)

        sum_l_dp = self.W_left(h)
        sum_r_dp = self.W_right(h)

        # compute the ratios for continuous binary tree
        n_children = h.shape[1]
        if n_children == 1:
            binary_right = th.tensor([0.5])
            binary_left = th.tensor([0.5])
        else:
            step = 1 / (n_children - 1)
            binary_right = th.arange(0, 1 + step, step)
            binary_left = 1 - binary_right
        sum_l = th.einsum("ijk,j->ik", sum_l_dp, binary_left)
        sum_r = th.einsum("ijk,j->ik", sum_r_dp, binary_right)
        sum_h = sum_l + sum_r

        return {"h_sum": sum_h}

    def apply_node_func(self, nodes: NodeBatch) -> Dict[str, th.Tensor]:
        new_h = self.W_top(nodes.data["h"])
        if "h_sum" in nodes.data:
            new_h += nodes.data["h_sum"]
        return {"new_h": new_h}

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        dgl.prop_nodes_topo(
            g, self.message_func, self.reduce_func, apply_node_func=self.apply_node_func
        )
        g.ndata["h"] = g.ndata["new_h"]
        del g.ndata["new_h"]
        return g


class TreeLayerNorm(nn.Module):
    def __init__(self) -> None:
        super(TreeLayerNorm, self).__init__()

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        g.ndata["h"] = nn.LayerNorm(g.ndata["h"].shape[1])(g.ndata["h"])
        return g


class TreeActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super(TreeActivation, self).__init__()
        self.activation = activation

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        g.ndata["h"] = self.activation(g.ndata["h"])
        return g


class DynamicPooling(nn.Module):
    def __init__(self, readout: ReadoutType):
        super(DynamicPooling, self).__init__()
        self.readout = readout

    def forward(self, g: dgl.DGLGraph) -> th.Tensor:
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, "h")
        elif self.readout == "max":
            hg = dgl.max_nodes(g, "h")
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, "h")
        elif self.readout == "terminal":
            if g.device == th.device("cpu"):
                out_op_inds_np = np.where(g.out_degrees().numpy() == 0)[0]
            else:
                out_op_inds_np = np.where(g.out_degrees().detach().cpu().numpy() == 0)[
                    0
                ]
            out_op_inds = th.tensor(out_op_inds_np, dtype=th.int32, device=g.device)
            hg = th.index_select(g.ndata["h"], 0, out_op_inds)
        else:
            raise NotImplementedError
        return hg


class TreeCNN(BaseGraphEmbedder):
    """TreeConv Embedder network.
    we extended the work to support other channels of inputs.
    """

    @dataclass
    class Params(BaseGraphEmbedder.Params):
        hidden_dim: int = 256
        """
        Size of the first channel of TreeConv, borrow the number for BAO
        https://github.com/learnedsystems/BaoForPostgreSQL/blob/28142c74903cbf0873b614b3b4f3bc49c5f84a1f/bao_server/net.py#L25
        """
        readout: ReadoutType = "max"
        """Readout type: using max pooling for BAO by default"""

    def __init__(self, net_params: Params) -> None:
        super().__init__(net_params)
        self.hidden_dim = net_params.hidden_dim
        self.net = nn.Sequential(
            TreeConvUnit(self.input_size, net_params.hidden_dim),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            TreeConvUnit(net_params.hidden_dim, net_params.hidden_dim // 2),
            TreeLayerNorm(),
            TreeActivation(nn.LeakyReLU()),
            TreeConvUnit(net_params.hidden_dim // 2, net_params.output_size),
            TreeLayerNorm(),
            DynamicPooling(net_params.readout),
        )

    def forward(self, g: dgl.DGLGraph) -> th.Tensor:  # type: ignore[override]
        g.ndata["h"] = self.concatenate_op_features(g)
        return self.net(g)
