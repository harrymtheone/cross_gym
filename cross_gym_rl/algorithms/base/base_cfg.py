from __future__ import annotations

from dataclasses import MISSING

from cross_gym.utils import configclass
from . import AlgorithmBase


@configclass
class AlgorithmBaseCfg:
    class_type: type[AlgorithmBase] = MISSING
