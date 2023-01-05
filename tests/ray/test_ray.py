import unittest
from core.ray import Ray
from core.constants import DIM
import numpy as np

from tests.ray.helper import rand_ray, repeat


class MyTestCase(unittest.TestCase):
    @repeat()
    def test_ray_init(self):
        ray = Ray.create()
        self.assertTrue(np.all(np.isnan(ray["o"])))
        self.assertTrue(np.all(np.isnan(ray["d"])))

    @repeat()
    def test_ray_init2(self):
        o, d, ray = rand_ray()

        np.testing.assert_array_equal(d, ray["d"])
        np.testing.assert_array_equal(o, ray["o"])

    @repeat()
    def test_ray_init3(self):
        d = np.random.rand(1, DIM)
        d_n = d / np.linalg.norm(d)
        o = np.random.rand(1, DIM)

        ray = Ray.create(o, d)

        np.testing.assert_allclose(d_n, ray["d"])
        np.testing.assert_allclose(o, ray["o"])

    @repeat()
    def test_ray_at(self):
        o, d, ray = rand_ray()
        t = np.random.rand()

        at = d*t+o

        np.testing.assert_allclose(at, Ray.at(ray, t))

    @repeat()
    def test_ray_components(self):
        o, d, ray = rand_ray()

        for i in range(DIM):
            np.testing.assert_array_equal(o[:, i:i+1], ray[f"o{i}"])
            np.testing.assert_array_equal(d[:, i:i+1], ray[f"d{i}"])

if __name__ == '__main__':
    unittest.main()
