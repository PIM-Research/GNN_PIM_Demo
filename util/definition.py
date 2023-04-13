from enum import Enum


class DropMode(Enum):
    ORIGINAL = 1
    GLOBAL = 2
    LOCAL = 3


class ClusterAlg(Enum):
    DBSCAN = 1
    K_MEANS = 2
