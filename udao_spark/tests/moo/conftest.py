import numpy as np
import pytest

from udao_spark.optimizer.moo_algos.dag_optimization import DAGOpt
from udao_spark.optimizer.moo_algos.hierarchical_moo_with_constraints import (
    Hierarchical_MOO_with_Constraints,
)

n_subQs = 3
seed = 0
np.random.seed(seed)
# np.random.randint(lower,upper,(row,colume))
theta_c_samples = np.random.randint(4, 20, (10, 2))
theta_p_samples = np.random.randint(20, 40, (2, 3))
theta_s_samples = np.random.randint(1, 4, (1, 2))

obj_values_all_subQs_all_theta_c = np.array(
    [[25, 20], [23, 23], [23, 25], [22, 24], [21, 26], [27, 27], [28, 21], [25, 29]]
)
configurations_all_subQs_all_theta_c = np.array(
    [
        [9, 4, 26, 27, 34, 1, 2],
        [9, 4, 37, 25, 33, 1, 2],
        [12, 16, 37, 25, 33, 1, 2],
        [9, 4, 26, 27, 34, 1, 2],
        [12, 16, 37, 25, 33, 1, 2],
        [9, 4, 26, 27, 34, 1, 2],
        [9, 4, 37, 25, 33, 1, 2],
        [12, 16, 26, 27, 34, 1, 2],
    ]
)
indices_all_subQs_all_theta_c = np.array(
    [
        [0, 1, 0],
        [0, 1, 1],
        [0, 8, 0],
        [1, 1, 0],
        [1, 8, 0],
        [2, 1, 0],
        [2, 1, 1],
        [2, 8, 0],
    ]
)
dag_opt_algorithm = "GD"
dag_opt_runmode = "naive"


@pytest.fixture
def hmooc() -> Hierarchical_MOO_with_Constraints:
    hmooc = Hierarchical_MOO_with_Constraints(
        n_subQs=n_subQs,
        theta_c_samples=theta_c_samples,
        theta_p_samples=theta_p_samples,
        theta_s_samples=theta_s_samples,
        seed=seed,
    )
    return hmooc


@pytest.fixture
def dag_opt() -> DAGOpt:
    dag_opt = DAGOpt(
        obj_values_all_subQs=obj_values_all_subQs_all_theta_c,
        config_all_subQs=configurations_all_subQs_all_theta_c,
        indices_all_subQs=indices_all_subQs_all_theta_c,
        algorithm=dag_opt_algorithm,
        runmode=dag_opt_runmode,
    )
    return dag_opt
