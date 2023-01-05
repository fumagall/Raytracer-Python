from abc import ABC, abstractmethod
from numpy import dtype, nan, zeros, copy, double, inner, sqrt, sum, cross, sign, ones
from numpy.linalg import norm

from core.ray import Ray
from core.constants import DIM, V_TYPE


class Record:
    TYPE = dtype({
        "p": (V_TYPE, 0),
        **{"p" + str(i): (("d", (1,)), 0 * DIM + 8 * i) for i in range(DIM)},
        "n": (V_TYPE, 8 * DIM),
        **{"n" + str(i): (("d", (1,)), 8 * DIM + 8 * i) for i in range(DIM)},
        "t": (("d", (1,)), 8 * DIM * 2),
    })

    @classmethod
    def create(cls, position=(nan, nan, nan), normal=(nan, nan, nan), t=nan, dim=1) -> TYPE:
        rec = zeros(dim, dtype=cls.TYPE)
        rec["p"] = copy(position)
        rec["n"] = copy(normal)
        rec["t"] = t
        return rec


class Hittable(ABC):

    @abstractmethod
    def hit(self, ray: Ray.TYPE, t_min: double, t_max: double, hit_record: Record) -> bool:
        pass


class Sphere(Hittable):
    def __init__(self, center, radius, color=(ones(3)*255)):
        self.center = center
        self.radius = radius
        self.color = color

    def hit(self, ray: Ray.TYPE, t_min: double, t_max: double, hit_record: Record.TYPE) -> bool:
        # this code was taken from https://raytracing.github.io/books/RayTracingInOneWeekend.html#surfacenormalsandmultipleobjects/anabstractionforhittableobjects

        oc = ray["o"] - self.center
        a = sum(ray["d"] ** 2, axis=-1)
        half_b = inner(oc, ray["d"])
        c = sum(oc ** 2) - self.radius ** 2

        discriminant = half_b ** 2 - a * c

        if discriminant < 0:
            return False

        sqrtd = sqrt(discriminant)

        root = (-half_b - sqrtd) / a
        if root < t_min or t_max < root:
            root = (-half_b + sqrtd) / a
            if root < t_min or t_max < root:
                return False

        hit_record["t"] = root
        hit_record["p"] = Ray.at(ray, root)
        hit_record["n"] = (hit_record["p"] - self.center) / self.radius

        return True

class Wall(Hittable):
    def __init__(self, position, vec1, vec2):
        self.position = position
        self.d1 = norm(vec1)
        self.e1 = vec1 / self.d1

        self.d2 = norm(vec2)
        self.e2 = vec2 / self.d2

        self.n = cross(self.e1, self.e2)

    def hit(self, ray: Ray.TYPE, t_min: double, t_max: double, hit_record: Record.TYPE) -> bool:
        orthogonal_distance_to_origin = inner(self.n, ray["d"])
        if all(orthogonal_distance_to_origin == 0):
            return False

        dist = inner((ray["o"]-self.position), self.n)

        alpha = -1*dist/orthogonal_distance_to_origin

        if alpha < t_min or alpha > t_max:
            return False

        intersection = ray["d"] * alpha + ray["o"] - self.position
        inner_d1_intersection = inner(intersection, self.e1)


        if 0 < inner_d1_intersection[0] and inner_d1_intersection[0] < self.d1:
            inner_d2_intersection = inner(intersection, self.e2)
            if 0 < inner_d2_intersection[0] and inner_d2_intersection[0] < self.d2:
                hit_record["t"] = alpha
                hit_record["p"] = Ray.at(ray, alpha)
                hit_record["n"] = -self.n * sign(orthogonal_distance_to_origin)

                return True

        return False

class Cube(Hittable):
    def hit(self, ray: Ray.TYPE, t_min: double, t_max: double, hit_record: Record) -> bool:
        t = t_max
        for wall in self.walls:
            tmp_rec = Record.create()  # why cant I change the line from inside the loop to outside?
            was_hit = wall.hit(ray, t_min, t_max, tmp_rec)

            if was_hit and all(tmp_rec["t"] < t):
                t = tmp_rec["t"]
                hit_record["p"] = copy(tmp_rec["p"])
                hit_record["n"] = copy(tmp_rec["n"])
                hit_record["t"] = copy(tmp_rec["t"])

        if t == t_max:
            return False
        else:
            return True

    def __init__(self, position, vec1, vec2):
        self.position = position
        self.d = norm(vec1)
        self.e1 = vec1/self.d
        self.v1 = vec1

        self.e2 = vec2 / self.d
        self.v2 = vec2

        self.en = cross(self.e1, self.e2)
        self.n = self.en*self.d

        self.walls = [
            Wall(position, self.v1, self.v2),
            #Wall(position + self.v1, self.n, self.v2),
            Wall(position + self.v2, self.v1, self.n),
            Wall(position + self.v1 + self.n, -self.v1, self.v2),
            Wall(position, self.n, self.v1),
            #Wall(position + self.v2, self.v1, self.n),
            #Wall(position + self.n, - self.n, self.v2),

            Wall(position + self.v1, self.n, self.v2),
            Wall(position + self.n, self.v1, self.v2),
        ]
