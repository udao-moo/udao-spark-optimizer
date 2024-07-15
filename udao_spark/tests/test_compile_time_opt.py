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
# Created at 14/02/2024

import numpy as np
from udao.optimization.utils.moo_utils import is_pareto_efficient


def test_is_pareto_efficient() -> None:
    po_objs = np.array(
        [
            [1.1, 2.8],
            [1.3, 2.5],
            [1.4, 2.3],
            [1.5, 2.4],
        ],
        dtype=object,
    )

    ret = is_pareto_efficient(po_objs)
    assert all(ret == [True, True, True, False])
