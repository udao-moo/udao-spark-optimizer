from dataclasses import dataclass
from typing import Dict, Tuple

import dgl
import torch as th
import torch.nn as nn
from dgl.udf import EdgeBatch, NodeBatch
from udao.model import BaseGraphEmbedder


class NeuralUnit(nn.Module):
    def __init__(
        self,
        node_type_id: int,
        input_dim: int,
        num_layers: int = 5,
        hidden_size: int = 128,
        output_size: int = 32,
    ):
        """
        Initialize the Neural Unit
        """
        super(NeuralUnit, self).__init__()
        self.node_type_id = node_type_id
        self.num_layers = num_layers  # num of hidden layers
        self.dense_block = self.build_block(
            num_layers, hidden_size, output_size, input_dim
        )

    def build_block(
        self, num_layers: int, hidden_size: int, output_size: int, input_dim: int
    ) -> nn.Sequential:
        """Construct a block consisting of linear Dense layers.
        Parameters:
            num_layers  (int)
            hidden_size (int)           -- the number of channels in the conv layer.
            output_size (int)           -- size of the output layer
            input_dim   (int)           -- input size, depends on each node_type
            norm_layer                  -- normalization layer
        Returns a conv block (with a conv layer, a normalization layer,
        and a non-linearity layer (ReLU))
        """
        assert num_layers >= 2
        dense_block = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for i in range(num_layers - 2):
            dense_block += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        dense_block += [nn.Linear(hidden_size, output_size)]
        return nn.Sequential(*dense_block)

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward function"""
        x = self.dense_block(x)
        lat = th.index_select(x, 1, th.zeros(1, dtype=th.int32, device=x.device))
        return lat, x


class QPPNet(BaseGraphEmbedder):
    """QPPNet"""

    @dataclass
    class Params(BaseGraphEmbedder.Params):
        num_layers: int
        hidden_size: int
        op_node2id: Dict[str, int]

    def __init__(self, net_params: Params) -> None:
        super().__init__(net_params)

        nu_dict, bn_dict = {}, {}
        self.input_nt_ids = {net_params.op_node2id["LogicalRelation"]}
        for nt, nt_id in net_params.op_node2id.items():
            if nt == "LogicalRelation":
                input_dim = self.input_size
            else:
                input_dim = self.input_size + net_params.output_size
            nu_dict[str(nt_id)] = NeuralUnit(
                node_type_id=nt_id,
                input_dim=input_dim,
                num_layers=net_params.num_layers,
                hidden_size=net_params.hidden_size,
                output_size=net_params.output_size,
            )
            bn_dict[str(nt_id)] = nn.BatchNorm1d(net_params.output_size)
        self.input_nt_id = net_params.op_node2id["LogicalRelation"]
        self.nu_dict = nn.ModuleDict(nu_dict)
        self.bn_dict = nn.ModuleDict(bn_dict)
        self.output_size = net_params.output_size

    def forward(self, g: dgl.DGLGraph) -> th.Tensor:  # type: ignore[override]
        n = g.num_nodes()
        g.ndata["feat"] = self.concatenate_op_features(g)
        g.ndata["children_outs"] = th.zeros((n, self.output_size)).to(g.device)
        g.ndata["out"] = th.zeros((n, self.output_size)).to(g.device)
        g.ndata["lat_hat"] = th.zeros((n, 1)).to(g.device)
        dgl.prop_nodes_topo(
            g, self.message_func, self.reduce_func, apply_node_func=self.apply_node_func
        )
        out_op_inds = th.where(g.out_degrees().detach() == 0)[0]
        lat_hats = th.exp(th.index_select(g.ndata["lat_hat"], 0, out_op_inds))
        return lat_hats

    def message_func(self, edges: EdgeBatch) -> Dict[str, th.Tensor]:
        return {"out": edges.src["out"]}

    def reduce_func(self, nodes: NodeBatch) -> Dict[str, th.Tensor]:
        return {"out": th.sum(nodes.mailbox["out"], 1)}

    def apply_node_func(self, nodes: NodeBatch) -> Dict[str, th.Tensor]:
        children_outs = nodes.data["children_outs"]
        n_rows = children_outs.shape[0]
        batch_nt_id = nodes.data["op_gid"]
        batch_feat = nodes.data["feat"]
        device = batch_feat.device

        unq_nt_ids = batch_nt_id.detach().unique()
        out = th.zeros_like(children_outs).to(device)
        lat_hat = th.zeros(n_rows, 1).to(device)
        for nt_id_tensor in unq_nt_ids:
            nt_id = nt_id_tensor.item()
            row_inds = th.where(batch_nt_id == nt_id)[0]
            nt_feat = th.index_select(batch_feat, 0, row_inds)
            cat_feat_list = [nt_feat]
            if nt_id not in self.input_nt_ids:
                nt_children_outs = th.index_select(children_outs, 0, row_inds)
                cat_feat_list.append(nt_children_outs)
            nt_input = th.cat(cat_feat_list, dim=1)
            nt_lat, nt_out = self.nu_dict[str(nt_id)].forward(nt_input)
            if nt_out.shape[0] > 1:
                nt_out = self.bn_dict[str(nt_id)](nt_out)
            out.index_add_(0, row_inds, nt_out)
            lat_hat.index_add_(0, row_inds, nt_lat)
        return {"out": out, "lat_hat": lat_hat}
