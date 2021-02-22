import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os


results = open('../BBoxes/train32pts_NoOverlap.csv', 'r').readlines()


image_to_boxes = {}
for line in results:
    items = line.split(',')
    if items[0] in image_to_boxes:
        image_to_boxes[items[0]].append(items[1:7])
    else:
        image_to_boxes[items[0]] = [items[1:7]]

fig, ax = plt.subplots(1)
fig.set_size_inches(10, 6)

for key in image_to_boxes:

    im = np.array(Image.open(key), dtype=np.uint8)
    ax.imshow(im)
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for box in image_to_boxes[key]:
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        rect = patches.Rectangle((x0, y0), (x1 - x0), (y1 - y0), linewidth='1', edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    plt.show(block=False)
    plt.pause(0.5)
    ax.cla()