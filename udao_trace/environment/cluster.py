from typing import List

from ..utils import ClusterName


class Cluster:
    def __init__(
        self,
        name: ClusterName,
    ):
        self.workers = self._get_workers(name)

    def _get_workers(self, name: ClusterName) -> List[str]:
        if name == ClusterName.HEX1:
            return ["node2", "node3", "node4", "node5", "node6"]
        elif name == ClusterName.HEX2:
            return ["node8", "node9", "node10", "node11", "node12"]
        elif name == ClusterName.HEX3:
            return ["node14", "node15", "node16", "node17", "node18"]
        else:
            raise ValueError(f"{name} is not supported")
