import math
import sys
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot(self, state_left, state_bricks):
    xx = np.arange(0, self.grid_width)
    yy = np.arange(0, self.grid_height)

    ax = plt.subplot(111, aspect='equal')
    for xi in xx:
        for yi in yy:
            if np.reshape(state_bricks, newshape=(self.grid_height, self.grid_width // 3))[
                int(yi), int(xi) // 3] == 1:
                sq = patches.Rectangle((xi, yi), .9, .9, fill=True, color='black')
                ax.add_patch(sq)
            if state_left[0] == int(xi) and state_left[1] == int(yi):
                sq = patches.Rectangle((xi, yi), 1, 1, fill=True, color='red')
                ax.add_patch(sq)

            if 0 <= xi - state_left[4] <= 4 and int(yi) == 0:
                sq = patches.Rectangle((xi, yi), 1, 1, fill=True, color='green')
                ax.add_patch(sq)

    ax.relim()
    ax.autoscale_view()
    plt.axis('off')
    plt.show()


d = dict()


for i in range(1,sys.maxsize):
    key = str(i)
    d[key] = key
    if math.log2(i) % 1 == 0: 
        time_start = time.perf_counter()
        value = d[key]
        time_taken = time.perf_counter() - time_start
        print(time_taken*1000*1000, i)