from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..utils import ScaleTypes, VarTypes
from ..utils.logging import logger
from .conf import Conf
from .knob_meta import KnobNumeric


class SparkConf(Conf):
    @staticmethod
    def _compute_k6(dop: KnobNumeric, k6: KnobNumeric) -> int:
        assert k6 in (0, 1), f"k6 value {k6} is not 0 or 1"
        k6_numeric = (
            (dop - 1 if dop <= 200 else 200)
            if k6 == 1
            else (dop + 1 if dop > 200 else 200)
        )
        return int(k6_numeric)

    @staticmethod
    def _inverse_compute_k6(dop: int, k6_numeric: int) -> int:
        k6 = 1 if k6_numeric < dop else 0
        return k6

    def _prepare_construction(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        knob_list = self.knob_list
        log_mask = np.array([k.scale == ScaleTypes.LOG for k in knob_list])
        linear_mask = np.array([k.scale == ScaleTypes.LINEAR for k in knob_list])
        ctype_int_mask = np.array(
            [
                k.ctype in (VarTypes.INT, VarTypes.CATEGORY, VarTypes.BOOL)
                for k in knob_list
            ]
        )
        ktype_int_mask = np.array(
            [
                k.ktype in (VarTypes.INT, VarTypes.CATEGORY, VarTypes.BOOL)
                for k in knob_list
            ]
        )
        factors = np.array([k.factor for k in knob_list]).astype(float)
        bases = np.array([k.base for k in knob_list]).astype(float)
        return log_mask, linear_mask, ctype_int_mask, ktype_int_mask, factors, bases

    def construct_configuration(self, conf_denorm: np.ndarray) -> np.ndarray:
        (
            log_mask,
            linear_mask,
            ctype_int_mask,
            _,
            factors,
            bases,
        ) = self._prepare_construction()
        # convert from numeric values to str parameter values,
        # with dependency resolved and unit added
        assert (len(conf_denorm) > 0) and (
            len(conf_denorm[0]) == len(self.knob_list)
        ), "number of columns of conf_denorm as a np.ndarray should match knob_list"
        conf_denorm[:, log_mask] = (
            np.power(bases[log_mask], conf_denorm[:, log_mask]) * factors[log_mask]
        )
        conf_denorm[:, linear_mask] *= factors[linear_mask]
        conf_denorm[:, ctype_int_mask] = np.round(conf_denorm[:, ctype_int_mask])
        conf_np = conf_denorm.copy()
        # a special transformation for k2
        conf_np[:, 1] = conf_np[:, 0] * conf_np[:, 1]
        # a special transformation for k4
        conf_np[:, 3] = conf_np[:, 0] * conf_np[:, 2] * conf_np[:, 3]
        # a special transformation for k6
        # k6: set spark.shuffle.sort.bypassMergeThreshold <= dop when True
        #     set spark.shuffle.sort.bypassMergeThreshold > dop when False
        conf_np[:, 5] = np.apply_along_axis(
            lambda c: self._compute_k6(dop=c[3], k6=c[5]), axis=1, arr=conf_np
        )
        # fixme: not handling categorical knobs
        return np.vectorize(lambda c, k: self._add_unit(c, k))(conf_np, self.knob_list)

    def deconstruct_configuration(self, conf: np.ndarray) -> np.ndarray:
        (
            log_mask,
            linear_mask,
            _,
            ktype_int_mask,
            factors,
            bases,
        ) = self._prepare_construction()
        conf_np = np.vectorize(lambda c, k: self._drop_unit(c, k))(conf, self.knob_list)
        conf_np[:, 5] = np.apply_along_axis(
            lambda c: self._inverse_compute_k6(dop=c[3], k6_numeric=c[5]),
            axis=1,
            arr=conf_np,
        )
        conf_np[:, 3] = conf_np[:, 3] / conf_np[:, 0] / conf_np[:, 2]
        conf_np[:, 1] = conf_np[:, 1] / conf_np[:, 0]
        conf_denorm = conf_np.copy()
        conf_denorm[:, linear_mask] /= factors[linear_mask]
        conf_denorm[:, log_mask] = np.log(
            conf_denorm[:, log_mask] / factors[log_mask]
        ) / np.log(bases[log_mask])
        conf_denorm[:, ktype_int_mask] = np.round(conf_denorm[:, ktype_int_mask])
        return conf_denorm

    def get_lhs_configurations(self, n_samples: int, seed: int) -> pd.DataFrame:
        assert n_samples > 1
        sampler = qmc.LatinHypercube(d=self.knob_num, seed=seed)
        conf_norm = sampler.random(n_samples)
        conf = self.construct_configuration_from_norm(conf_norm)
        unique_rows, counts = np.unique(conf, axis=0, return_counts=True)
        duplicated_rows = unique_rows[counts > 1]
        # Display the result
        if len(duplicated_rows) > 0:
            logger.warn("Duplicated Rows:", duplicated_rows)
        else:
            logger.debug("No duplicated rows.")
        np.random.seed(seed)
        np.random.shuffle(unique_rows)
        conf_df = pd.DataFrame(unique_rows, columns=self.knob_names)
        conf_df["conf_sign"] = conf_df.apply(lambda x: self.conf2sign(x), axis=1)
        conf_df.set_index("conf_sign", inplace=True)
        logger.debug(f"finished generated {len(conf_df)} configurations via LHS")
        return conf_df
