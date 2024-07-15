from typing import List, Optional, Union

from udao_trace.utils import ScaleTypes, VarTypes

KnobNumeric = Union[float, int]


class KnobMeta(object):
    @staticmethod
    def _get_vartype(type: str) -> VarTypes:
        if type == "int":
            return VarTypes.INT
        elif type == "bool":
            return VarTypes.BOOL
        elif type == "category":
            return VarTypes.CATEGORY
        elif type == "float":
            return VarTypes.FLOAT
        else:
            raise Exception(f"unknown type {type}")

    def __init__(
        self,
        kid: str,
        name: str,
        ktype: str,  # knob type
        ctype: str,  # construction type
        unit: Optional[str],
        kmin: Optional[Union[int, float]],
        kmax: Optional[Union[int, float]],
        scale: Optional[ScaleTypes],
        factor: Optional[Union[int, float]],
        base: Optional[int],
        categories: Optional[List[str]],
        default: str,
        desc: str,
    ):
        self.id = kid
        self.name = name
        self.ktype = self._get_vartype(ktype)
        self.ctype = self._get_vartype(ctype)
        self.unit = unit
        self.min = kmin
        self.max = kmax
        self.scale = (
            ScaleTypes.LINEAR if scale is None or scale == "linear" else ScaleTypes.LOG
        )
        self.factor = factor if factor is not None else 1
        self.base = base
        self.categories = categories
        self.default = default
        self.desc = desc

        if self.ktype in (VarTypes.INT, VarTypes.FLOAT):
            assert (
                self.min is not None and self.max is not None
            ), f"min/max value of {self.name} is not set"
            assert (
                self.min <= self.max
            ), f"min value {self.min} is larger than max value {self.max}"
            assert self.factor is not None, f"factor of {self.name} is not set"
        elif self.ktype == VarTypes.CATEGORY:
            assert (
                self.categories is not None and len(self.categories) >= 1
            ), f"categories of {self.name} is not set"

        if self.ktype == VarTypes.BOOL:
            self.min = 0
            self.max = 1
        elif self.ktype == VarTypes.CATEGORY:
            assert categories is not None
            self.min = 0
            self.max = len(categories) - 1
