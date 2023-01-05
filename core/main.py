from numpy import zeros

from core.Sensor import Camera
from core.hittable import Sphere
from core.world import World

import matplotlib.pyplot as plt

s = Sphere((0,0,0), 1)

world = World([s])

view = zeros([480//4,640//4, 3], dtype="int")
cam = Camera(view, (0,0,-5), (0,0,-5), 2, (1,0,0), (0,1,0))

cam.render()

plt.imshow(cam.view)