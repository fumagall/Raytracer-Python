# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from numpy import zeros, array, sin, cos

from core.Sensor import Camera
from core.hittable import Sphere, Wall, Cube
from core.world import World

import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    a = lambda *x: array([x], dtype="double")

    world = World([
        #Sphere((0, 0, -2), 0.5),
        Cube(a(0,0,-2), a(0.5,-0.1,0.1), a(-0.1,0.5,0.1)),
        Sphere((0, 0, 2), 0.5),
        Wall(a(-1,-1,-0.5), a(2,0,1), a(0, 2, 1))
    ])

    view = zeros([480 // 8, 640 // 8, 1], dtype="int")
    cam = Camera(view, a(0, 0, -10), a(0, 0, -5), 2, a(1, 0, 0), a(0, 1, 0), world)

    rot10 = lambda x, ac=1 : (array([
        [1, 0, 0],
        [0, cos(0.1 * ac), -sin(0.1 * ac)],
        [0, sin(0.1 * ac), cos(0.1 * ac),],

    ]) @ x[0])[None,:]

    fig, ax = plt.subplots()
    cam.render()


    def animate(i):
        global cam

        ax.clear()
        ax.imshow(cam.view)
        ax.set_xlim([0, cam.h])
        ax.set_ylim([0, cam.w])

        cam = Camera(view, rot10(cam.pos), a(0, 0, -5), 2, rot10(cam.dx), rot10(cam.dy), world)
        cam.render()
        print(i)


    ani = FuncAnimation(fig, animate, frames=62, interval=10, repeat=False)
    plt.show()
    FFwriter = animation.FFMpegWriter(fps=20)
    ani.save('animation2.mp4', writer=FFwriter)

    print("finsihed")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
