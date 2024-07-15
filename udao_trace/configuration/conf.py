import math
import re
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Union

import numpy as np

from ..utils import JsonHandler, ScaleTypes, VarTypes
from .knob_meta import KnobMeta, KnobNumeric

unit_to_multiplier = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}


class Conf(ABC):
    def __init__(self, meta_file: str):
        knob_meta = JsonHandler.load_json(meta_file)

        self.knob_list = [
            KnobMeta(
                k["id"],
                k["name"],
                k["type"],
                k["construction_type"],
                k["unit"],
                k["min"],
                k["max"],
                k["scale"],
                k["factor"],
                k["base"],
                k["categories"],
                k["default"],
                k["desc"],
            )
            for k in knob_meta
        ]
        self.knob_dict_by_id = {k.id: k for k in self.knob_list}
        self.knob_dict_by_name = {k.name: k for k in self.knob_list}
        self.knob_ids = [k.id for k in self.knob_list]
        self.knob_names = [k.name for k in self.knob_list]
        self.knob_num = len(self.knob_list)
        self.knob_min = [k.min for k in self.knob_list]
        self.knob_max = [k.max for k in self.knob_list]

    @staticmethod
    def _construct_knob(k_denorm: float, k_meta: KnobMeta) -> KnobNumeric:
        if k_meta.scale == ScaleTypes.LINEAR:
            k_value = k_denorm * k_meta.factor
        elif k_meta.scale == ScaleTypes.LOG:
            assert k_meta.base is not None
            k_value = math.pow(k_meta.base, k_denorm) * k_meta.factor
        else:
            raise Exception(f"unknown scale type {k_meta.scale}")
        if k_meta.ctype in (VarTypes.INT, VarTypes.CATEGORY, VarTypes.BOOL):
            k_value = int(round(k_value))
        return k_value

    @staticmethod
    def _deconstruct_knob(k_value: KnobNumeric, k_meta: KnobMeta) -> float:
        if k_meta.scale == ScaleTypes.LINEAR:
            k_denorm = k_value / k_meta.factor
        elif k_meta.scale == ScaleTypes.LOG:
            assert k_meta.base is not None
            k_denorm = math.log(k_value / k_meta.factor, k_meta.base)
        else:
            raise Exception(f"unknown scale type {k_meta.scale}")
        if k_meta.ktype in (VarTypes.INT, VarTypes.CATEGORY, VarTypes.BOOL):
            k_denorm = int(round(k_denorm))
        return k_denorm

    @staticmethod
    def _add_unit(k_value: KnobNumeric, k_meta: KnobMeta) -> str:
        if k_meta.ctype == VarTypes.BOOL:
            assert k_value in (0, 1), f"bool value {k_value} is not 0 or 1"
            return "true" if k_value == 1 else "false"
        return f"{k_value:g}" if k_meta.unit is None else f"{k_value:g}{k_meta.unit}"

    @staticmethod
    def _drop_unit(k_with_unit: str, k_meta: KnobMeta) -> float:
        if k_meta.ctype == VarTypes.BOOL:
            assert k_with_unit.lower() in (
                "true",
                "false",
            ), f"bool value {k_with_unit} is not true or false"
            return 1.0 if k_with_unit.lower() == "true" else 0.0
        if k_meta.unit is not None:
            if k_with_unit.endswith(k_meta.unit):
                k_with_unit = k_with_unit[: -len(k_meta.unit)]
            elif k_with_unit.upper().endswith(k_meta.unit[-1:].upper()):
                splits = re.split("(\d+)", k_with_unit)
                if len(splits) == 3:
                    k_with_unit = str(
                        float(splits[1])
                        * unit_to_multiplier[splits[2].upper()]
                        / unit_to_multiplier[k_meta.unit.upper()]
                    )
                else:
                    raise Exception(
                        f"unit {k_meta.unit} cannot be parsed with {k_with_unit}"
                    )
            else:
                raise Exception(
                    f"unit {k_meta.unit} cannot be parsed with {k_with_unit}"
                )

        if k_meta.ctype in (VarTypes.INT, VarTypes.CATEGORY, VarTypes.FLOAT):
            return float(k_with_unit)
        else:
            raise Exception(f"unknown ctype {k_meta.ctype}")

    def get_default_conf(self, to_dict: bool = True) -> Union[Dict, List]:
        if to_dict:
            return {k.name: k.default for k in self.knob_list}
        else:
            return [k.default for k in self.knob_list]

    # def normalize(
    #     self, conf: Union[Iterable, List[float]]
    # ) -> Union[Iterable, List[float]]:
    #     ...

    def denormalize(
        self, conf_norm: np.ndarray, eps: float = float(np.finfo(float).eps)
    ) -> np.ndarray:
        if (conf_norm < 0).any() or (conf_norm > 1).any():
            raise Exception(
                f"conf_norm should be within [0, 1], got {conf_norm} instead"
            )
        conf_norm = np.clip(conf_norm, 0.0, 1.0 - eps)
        knob_list, knob_min, knob_max = (
            self.knob_list,
            np.array(self.knob_min),
            np.array(self.knob_max),
        )
        ktype_int_mask = np.array(
            [
                k.ktype in (VarTypes.INT, VarTypes.CATEGORY, VarTypes.BOOL)
                for k in knob_list
            ]
        )
        ktpye_float_mask = np.array([k.ktype == VarTypes.FLOAT for k in knob_list])
        assert (len(conf_norm) > 0) and (
            len(conf_norm[0]) == len(self.knob_list)
        ), "number of columns of conf_norm as a np.ndarray should match knob_list"
        x = conf_norm
        x[:, ktpye_float_mask] = knob_min[ktpye_float_mask] + x[:, ktpye_float_mask] * (
            knob_max[ktpye_float_mask] - knob_min[ktpye_float_mask]
        )
        x[:, ktype_int_mask] = knob_min[ktype_int_mask] + x[:, ktype_int_mask] * (
            knob_max[ktype_int_mask] - knob_min[ktype_int_mask] + 1
        )
        x[:, ktype_int_mask] = np.floor(x[:, ktype_int_mask])
        return x

    @abstractmethod
    def construct_configuration(self, conf_denorm: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def deconstruct_configuration(self, conf: np.ndarray) -> np.ndarray:
        pass

    def construct_configuration_from_norm(self, conf_norm: np.ndarray) -> np.ndarray:
        conf_denorm = self.denormalize(conf_norm)
        return self.construct_configuration(conf_denorm)

    @staticmethod
    def conf2sign(conf: Iterable[str]) -> str:
        """
        serialize a sequence of knob values into a string
        :param conf: a sequence of knob values (e.g., a list of knobs)
        :return: the sign of the knob values
        """
        return ",".join([c for c in conf])
