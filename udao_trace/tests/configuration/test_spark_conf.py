from pathlib import Path

import numpy as np
import pytest

from ...configuration import SparkConf


@pytest.fixture()
def sc() -> SparkConf:
    base_dir = Path(__file__).parent
    knob_meta_file = str(base_dir.parent / "assets/spark_configuration_aqe_on.json")
    return SparkConf(knob_meta_file)


EPS = 1e-6
CONG_NORM1 = [0.0] * 19
CONG_NORM2 = [1.0 - EPS] * 19
CONG_NORM3 = [
    0.0,
    0.0,
    1.0,
    0.0,
    0.4,
    0.0,
    1.0,
    0.4,
    0.4,
    0.2,
    0.0,
    1 / 32,
    23 / 48,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.2,
]

CONF_DENORM1 = [1, 1, 4, 1, 0, 0, 0, 50, 0, 1, 0, 0, 2, 0, 20, 0, 0, 5, 1]
CONF_DENORM2 = [5, 4, 16, 4, 5, 1, 1, 75, 5, 6, 32, 32, 50, 4, 80, 4, 4, 35, 6]
CONF_DENORM3 = [1, 1, 16, 1, 2, 0, 1, 60, 2, 2, 0, 1, 25, 2, 50, 2, 2, 20, 2]

CONF1 = [
    "1",
    "1g",
    "4",
    "4",
    "12m",
    "200",
    "false",
    "0.5",
    "16MB",
    "0.1",
    "0MB",
    "0MB",
    "16",
    "64MB",
    "2",
    "32MB",
    "1MB",
    "0.05",
    "512KB",
]
CONF2 = [
    "5",
    "20g",
    "16",
    "320",
    "384m",
    "200",
    "true",
    "0.75",
    "512MB",
    "0.6",
    "320MB",
    "320MB",
    "400",
    "1024MB",
    "8",
    "512MB",
    "16MB",
    "0.35",
    "3072KB",
]
CONF3 = [
    "1",
    "1g",
    "16",
    "16",
    "48m",
    "200",
    "true",
    "0.6",
    "64MB",
    "0.2",
    "0MB",
    "10MB",
    "200",
    "256MB",
    "5",
    "128MB",
    "4MB",
    "0.2",
    "1024KB",
]
HEADER = [
    "k1",
    "k2",
    "k3",
    "k4",
    "k5",
    "k6",
    "k7",
    "k8",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
    "s11",
]


class TestSparkConf:
    @pytest.mark.parametrize(
        "conf_norm, expected_conf_denorm",
        [
            (
                np.array([CONG_NORM1, CONG_NORM2, CONG_NORM3]),
                np.array([CONF_DENORM1, CONF_DENORM2, CONF_DENORM3]),
            )
        ],
    )
    def test_denormalize(
        self,
        sc: SparkConf,
        conf_norm: np.ndarray,
        expected_conf_denorm: np.ndarray,
    ) -> None:
        assert np.array_equal(sc.denormalize(conf_norm), expected_conf_denorm)

    @pytest.mark.parametrize(
        "conf_denorm, expected_conf",
        [
            (
                np.array([CONF_DENORM1, CONF_DENORM2, CONF_DENORM3]).astype(float),
                np.array([CONF1, CONF2, CONF3]).astype(str),
            )
        ],
    )
    def test_construct_configuration(
        self, sc: SparkConf, conf_denorm: np.ndarray, expected_conf: np.ndarray
    ) -> None:
        assert np.array_equal(sc.construct_configuration(conf_denorm), expected_conf)

    @pytest.mark.parametrize(
        "conf, expected_conf_denorm",
        [
            (
                np.array([CONF1, CONF2, CONF3]).astype(str),
                np.array([CONF_DENORM1, CONF_DENORM2, CONF_DENORM3]).astype(float),
            ),
        ],
    )
    def test_deconstruct_configuration(
        self, sc: SparkConf, conf: np.ndarray, expected_conf_denorm: np.ndarray
    ) -> None:
        conf_denorm = sc.deconstruct_configuration(conf)
        assert np.array_equal(conf_denorm, expected_conf_denorm)

    def test_knob_sign_with_default(self, sc: SparkConf) -> None:
        knob_sign_default = ",".join([str(k.default) for k in sc.knob_list])
        assert ",".join(CONF3) == knob_sign_default
