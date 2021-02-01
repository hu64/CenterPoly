import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import json


base_dir = '/store/datasets/cityscapes'
anno = json.load(open('../BBoxes/val.json', 'r'))

id_to_file = {}
for image in anno['images']:
    id_to_file[image['id']] = image['file_name']

results = json.load(open('/usagers2/huper/dev/CenterPoly/exp/cityscapes/polydet/hg_32pts/results.json', 'r'))
image_to_boxes = {}
for result in results:
    box = list(result['bbox'])
    box.append(result['score'])
    box += list(result['polygons'])
    if id_to_file[result['image_id']] in image_to_boxes:
        image_to_boxes[id_to_file[result['image_id']]].append(box)
    else:
        image_to_boxes[id_to_file[result['image_id']]] = [box]

fig, ax = plt.subplots(1)
fig.set_size_inches(10, 6)

count = 1
for key in sorted(image_to_boxes):

    im = np.array(Image.open(os.path.join(base_dir, key)), dtype=np.uint8)
    ax.imshow(im)
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for box in image_to_boxes[key]:
        if box is None:
            continue
        x0, y0, h, w = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        lw = float(box[4]) * 2
        rect = patches.Rectangle((x0, y0), h, w, linewidth=lw, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        points = []
        for i in range(6, len(box)-1, 2):
            points.append((box[i], box[i+1]))
        poly = patches.Polygon(points, linewidth=lw, edgecolor='y', facecolor='none')
        ax.add_patch(poly)

    plt.show(block=False)
    plt.pause(0.5)
    ax.cla()