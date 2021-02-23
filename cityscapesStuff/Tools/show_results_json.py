import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import json

TRESH = 0.25
base_dir = '/store/datasets/cityscapes'
anno = json.load(open('../BBoxes/val.json', 'r'))

id_to_file = {}
for image in anno['images']:
    id_to_file[image['id']] = image['file_name']

results_file = '/usagers2/huper/dev/CenterPoly/exp/cityscapes/polydet/gt_from_pts/results.json'
results = json.load(open(results_file, 'r'))
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
    for poly in image_to_boxes[key]:
        score = float(poly[0])
        label = int(poly[1])
        ec = ''
        if label == 0:
            ec = 'blue'
        elif label == 1:
            ec = 'purple'
        elif label == 2:
            ec = 'orange'
        elif label == 3:
            ec = 'olive'
        elif label == 4:
            ec = 'green'
        elif label == 5:
            ec = 'red'
        elif label == 6:
            ec = 'brown'
        elif label == 7:
            ec = 'gray'

        if score >= TRESH:
            lw = score * 2
            points = []
            for i in range(2, len(poly)-1, 2):
                points.append((poly[i], poly[i+1]))
            poly = patches.Polygon(points, linewidth=lw, edgecolor=ec, facecolor='none')
            ax.add_patch(poly)

    plt.show(block=False)
    # plt.savefig(os.path.join(os.path.dirname(results_file), 'image_examples', os.path.basename(key)))
    plt.pause(0.5)
    ax.cla()