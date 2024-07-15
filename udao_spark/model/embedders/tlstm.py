from dataclasses import dataclass
from typing import Dict

import dgl
import numpy as np
import torch as th
import torch.nn as nn
from dgl.udf import EdgeBatch, NodeBatch
from udao.model import BaseGraphEmbedder
from udao.model.embedders.graph_transformer import ReadoutType


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size: int, h_size: int):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges: EdgeBatch) -> Dict[str, th.Tensor]:
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes: NodeBatch) -> Dict[str, th.Tensor]:
        h_tild = th.sum(nodes.mailbox["h"], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = th.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_tild), "c": c}

    def apply_node_func(self, nodes: NodeBatch) -> Dict[str, th.Tensor]:
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * th.tanh(c)
        return {"h": h, "c": c}


class TreeLSTM(BaseGraphEmbedder):
    """TreeLSTM Embedder network.
    we extended the work to support other channels of inputs.
    """

    @dataclass
    class Params(BaseGraphEmbedder.Params):
        hidden_dim: int
        """Size of the hidden layers outputs."""
        dropout: float = 0.0
        """Dropout probability."""
        readout: ReadoutType = "mean"
        """Readout type: how the node embeddings are aggregated"""

    def __init__(self, net_params: Params) -> None:
        super().__init__(net_params)

        self.hidden_dim = net_params.hidden_dim
        self.embedding_h = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
        )
        self.dropout = nn.Dropout(net_params.dropout)
        self.readout = net_params.readout
        self.cell = ChildSumTreeLSTMCell(self.hidden_dim, self.hidden_dim)
        self.h2embed = nn.Sequential(
            nn.Linear(self.hidden_dim, net_params.output_size),
            nn.ReLU(),
            nn.BatchNorm1d(net_params.output_size),
        )

    def _embed(self, g: dgl.DGLGraph, h: th.Tensor) -> th.Tensor:
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        g : dgl.DGLGraph
            Tree for computation.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        # feed embedding
        h = self.embedding_h(h)

        g.ndata["iou"] = self.cell.W_iou(self.dropout(h))

        n = g.num_nodes()
        g.ndata["h"] = th.zeros((n, self.hidden_dim)).to(g.device)
        g.ndata["c"] = th.zeros((n, self.hidden_dim)).to(g.device)

        # propagate
        dgl.prop_nodes_topo(
            g,
            self.cell.message_func,
            self.cell.reduce_func,
            apply_node_func=self.cell.apply_node_func,
        )

        g.ndata["h_embeds"] = self.dropout(self.h2embed(g.ndata["h"]))

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, "h_embeds")
        elif self.readout == "max":
            hg = dgl.max_nodes(g, "h_embeds")
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, "h_embeds")
        elif self.readout == "terminal":
            if g.device == th.device("cpu"):
                out_op_inds_np = np.where(g.out_degrees().numpy() == 0)[0]
            else:
                out_op_inds_np = np.where(g.out_degrees().detach().cpu().numpy() == 0)[
                    0
                ]
            out_op_inds = th.tensor(out_op_inds_np, dtype=th.int32, device=g.device)
            hg = th.index_select(g.ndata["h_embeds"], 0, out_op_inds)
        else:
            raise NotImplementedError
        return hg

    def forward(self, g: dgl.DGLGraph) -> th.Tensor:  # type: ignore[override]
        h = self.concatenate_op_features(g)
        return self.normalize_embedding(self._embed(g, h))
