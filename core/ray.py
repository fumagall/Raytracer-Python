

from numpy import dtype, nan, zeros, copy
from numpy.linalg import norm
from core.constants import DIM, V_TYPE


class Ray:
    TYPE = dtype({
        "o": (V_TYPE, 0),
        **{"o" + str(i): (("d", (1,)), 0 * DIM + 8 * i) for i in range(DIM)},
        "d": (V_TYPE, 8 * DIM),
        **{"d" + str(i): (("d", (1,)), 8 * DIM + 8 * i) for i in range(DIM)}
    })

    @classmethod
    def create(cls, origin=(nan, nan, nan), direction=(nan, nan, nan), dim=1, normalize=True) -> TYPE:
        ray = zeros(dim, dtype=cls.TYPE)
        ray["o"] = copy(origin)
        ray["d"] = copy(direction)
        if normalize:
            ray["d"] = ray["d"]/norm(direction)

        return ray

    @staticmethod
    def at(ray, t) -> TYPE:
        return ray["d"] * t + ray["o"]