import unittest
from core.hittable import Record, Sphere
import numpy as np
from core.constants import DIM
from core.ray import Ray
from helper import rand_rec, rand_ray, repeat


class MyRecordTest(unittest.TestCase):
    def test_record_init(self):
        rec = Record.create()

        self.assertTrue(np.all(np.isnan(rec["p"])))
        self.assertTrue(np.all(np.isnan(rec["n"])))
        self.assertTrue(np.all(np.isnan(rec["t"])))

    @repeat()
    def test_record_init2(self):
        p, n, t, rec = rand_rec()

        np.testing.assert_array_equal(p, rec["p"])
        np.testing.assert_array_equal(n, rec["n"])
        np.testing.assert_array_equal(t, rec["t"])

    @repeat()
    def test_record_components(self):
        p, n, t, rec = rand_rec()

        for i in range(DIM):
            np.testing.assert_array_equal(n[:, i:i+1], rec[f"n{i}"])
            np.testing.assert_array_equal(p[:, i:i+1], rec[f"p{i}"])


class MySphereTest(unittest.TestCase):
    @repeat()
    def test_sphere_init(self):
        center = np.random.rand(1, DIM)
        radius = np.random.rand(1, 1)

        s = Sphere(center, radius)

        np.testing.assert_array_equal(center, s.center)
        np.testing.assert_array_equal(radius, s.radius)

    @repeat()
    def test_hit_sphere(self):
        center = np.zeros((1, DIM))
        radius = np.ones((1, 1))

        s = Sphere(center, radius)

        t_min = 0
        t_max = 10

        rec = Record.create()

        o = np.random.rand(1, DIM) * 2 - 1
        ray = Ray.create(o, o)

        was_hit = s.hit(ray, t_min, t_max, rec)

        self.assertEqual(was_hit, np.linalg.norm(o) <= 1, msg=[np.linalg.norm(o), ray])


class MyWallTest(unittest.TestCase):
    @repeat()
    def test_sphere_init(self):
        center = np.random.rand(1, DIM)
        radius = np.random.rand(1, 1)

        s = Sphere(center, radius)

        np.testing.assert_array_equal(center, s.center)
        np.testing.assert_array_equal(radius, s.radius)

    @repeat()
    def test_hit_sphere(self):
        center = np.zeros((1, DIM))
        radius = np.ones((1, 1))

        s = Sphere(center, radius)

        t_min = 0
        t_max = 10

        rec = Record.create()

        o = np.random.rand(1, DIM) * 2 - 1
        ray = Ray.create(o, o)

        was_hit = s.hit(ray, t_min, t_max, rec)

        self.assertEqual(was_hit, np.linalg.norm(o) <= 1, msg=[np.linalg.norm(o), ray])

if __name__ == '__main__':
    unittest.main()
