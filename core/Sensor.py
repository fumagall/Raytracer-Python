from abc import ABC, abstractmethod

from mkl_random import rand
from numpy import cross, inner
from numpy.linalg import norm

from core.constants import V_TYPE
from core.hittable import Record
from core.ray import Ray
from core.world import World

from numba import jit, prange


class Sensor(ABC):
    def __init__(self, position: V_TYPE, view_direction: V_TYPE, world: World):
        self.position = position
        self.view_direction = view_direction
        self.world = world

    def update_position(self, position: V_TYPE, view_direction: V_TYPE) -> None:
        self.position = position
        self.view_direction = view_direction

    @abstractmethod
    def render(self) -> Ray.TYPE:
        pass


class Camera(Sensor):
    def __init__(self, view, position, view_direction, distance, direction_x, direction_y, world):
        self.view = view
        self.w = view.shape[0]
        self.h = view.shape[1]
        self.d = distance
        self.dx = direction_x
        self.dy = direction_y
        self.pos = position
        self.ez = cross(self.dx, self.dy)
        super().__init__(position, view_direction, world)

    from multiprocessing import Pool

    def render(self):

        for x in range(self.w):
            for y in range(self.h):
                rec = Record.create()
                # print(self.ez)

                color = 0.0

                direction = self.d * self.ez + (x - self.w // 2 ) / self.w * self.dx + (
                                y - self.h // 2 ) / self.h * self.dy

                origin = self.pos + direction
                direction /= norm(direction)

                ray = Ray.create(origin, direction)

                if self.world.hit(ray, 0, 100, rec):
                    color = int(-255 * inner(rec["n"], ray["d"] / norm(ray["d"])))

                # print(norm(rec[Hittable.n]), norm(ray[Ray.d]))
                self.view[x, y, :] = color