import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import os
import json

TRESH = 0.1
base_dir = '/store/datasets/cityscapes'
anno = json.load(open('../BBoxes/val.json', 'r'))

id_to_file = {}
for image in anno['images']:
    id_to_file[image['id']] = image['file_name']

results_file = '/usagers2/huper/dev/CenterPoly/exp/cityscapes/polydet/hg_64pts_lossNormWithin_WeSu/results.json'
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
    #     if box is None:
    #         continue
    #     x0, y0, h, w = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        score = float(poly[0])
        if score >= TRESH:
            lw = score * 2
            # rect = patches.Rectangle((x0, y0), h, w, linewidth=lw, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            points = []
            for i in range(2, len(poly)-1, 2):
                points.append((poly[i], poly[i+1]))
            poly = patches.Polygon(points, linewidth=lw, edgecolor='y', facecolor='none')
            ax.add_patch(poly)

    plt.show(block=False)
    # plt.savefig(os.path.join(os.path.dirname(results_file), 'image_examples', os.path.basename(key)))
    plt.pause(0.5)
    ax.cla()