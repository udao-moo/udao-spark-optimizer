from typing import List

from ..utils import BenchmarkType


class Benchmark:
    def __init__(self, benchmark_type: BenchmarkType, scale_factor: int = 100):
        self.benchmark_type = benchmark_type
        self.scale_factor = scale_factor
        self.templates = self._get_templates(benchmark_type)
        self.template2id = {t: i for i, t in enumerate(self.templates)}

    def get_name(self) -> str:
        return self.benchmark_type.value

    def get_prefix(self) -> str:
        return f"{self.get_name()}{self.scale_factor}"

    def get_template_id(self, template: str) -> int:
        return self.template2id[template]

    def _get_templates(self, benchmark_type: BenchmarkType) -> List[str]:
        if benchmark_type == BenchmarkType.TPCH:
            return [str(i) for i in range(1, 23)]
        elif benchmark_type == BenchmarkType.TPCDS:
            return [
                t
                for t in "1 2 3 4 5 6 7 8 9 10 11 12 13 14a 14b 15 16 17 18 "
                "19 20 21 22 23a 23b 24a 24b 25 26 27 28 29 30 31 "
                "32 33 34 35 36 37 38 39a 39b 40 41 42 43 44 45 46 "
                "47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 "
                "64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 "
                "81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 "
                "98 99".split()
                if t not in ["61"]  # to be dropped due to parsing issue
            ]
        elif benchmark_type == BenchmarkType.TPCXBB:
            return [str(i) for i in range(1, 31)]
        else:
            raise ValueError(f"{benchmark_type} is not supported")
