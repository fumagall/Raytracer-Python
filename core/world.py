from typing import List

from numpy import double, copy

from core.hittable import Hittable, Record
from core.ray import Ray


class World(Hittable):
    def __init__(self, elements: List[Hittable]):
        self.elements = elements

    def get_elements(self):
        return self.elements

    def hit(self, ray: Ray.TYPE, t_min: double, t_max: double, hit_record: Record) -> bool:
        t = t_max

        for o in self.elements:
            tmp_rec = Record.create() #why cant I change the line from inside the loop to outside?
            was_hit = o.hit(ray, t_min, t_max, tmp_rec)

            if was_hit and all(tmp_rec["t"] < t):
                t = tmp_rec["t"]
                hit_record["p"] = copy(tmp_rec["p"])
                hit_record["n"] = copy(tmp_rec["n"])
                hit_record["t"] = copy(tmp_rec["t"])

        if t == t_max:
            return False
        else:
            return True