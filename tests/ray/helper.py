import numpy as np

from core.constants import DIM
from core.hittable import Record
from core.ray import Ray


TEST_LENGTH = 1000

def repeat():
    def repeatHelper(f):
        def callHelper(*args):
            for i in range(0, TEST_LENGTH):
                f(*args)

        return callHelper

    return repeatHelper

def rand_rec() -> [Record.TYPE["p"], Record.TYPE["n"], Record.TYPE["t"], Record.TYPE]:
    p = np.random.rand(1, DIM)
    n = np.random.rand(1, DIM)
    t = np.random.rand(1, 1)

    ray = Record.create(p, n, t)

    return p, n, t, ray


def rand_ray() -> [Ray.TYPE["o"], Ray.TYPE["d"], Ray.TYPE]:
    d = np.random.rand(1, DIM)
    d = d / np.linalg.norm(d)
    d = d.astype("double")
    o = np.random.rand(1, DIM)

    ray = Ray.create(o, d, normalize=False)

    return o, d, ray
